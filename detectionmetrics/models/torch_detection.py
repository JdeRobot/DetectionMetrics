from collections import defaultdict
import os
import time
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from tqdm.notebook import tqdm

from detectionmetrics.datasets import detection as dm_detection_dataset
from detectionmetrics.models import detection as dm_detection_model
from detectionmetrics.utils import detection_metrics as um


def data_to_device(
    data: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    device: torch.device,
) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Move detection input or target data (dict or list of dicts) to the specified device.

    :param data: Detection data (a single dict or list of dicts with tensor values)
    :type data: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    :param device: Device to move data to
    :type device: torch.device
    :return: Data with all tensors moved to the target device
    :rtype: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    """
    if isinstance(data, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}

    elif isinstance(data, list):
        return [
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in item.items()}
            for item in data
        ]

    else:
        raise TypeError(f"Expected a dict or list of dicts, got {type(data)}")


def get_data_shape(data: Union[torch.Tensor, tuple]) -> tuple:
    """Get the shape of the provided data

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :return: Data shape
    :rtype: Union[tuple, list]
    """
    if isinstance(data, tuple):
        return data[0].shape
    return data.shape


def get_computational_cost(
    model: Any,
    dummy_input: Union[torch.Tensor, tuple, list],
    model_fname: Optional[str] = None,
    runs: int = 30,
    warm_up_runs: int = 5,
) -> pd.DataFrame:
    """
    Get different metrics related to the computational cost of a model.

    :param model: TorchScript or PyTorch model (segmentation, detection, etc.)
    :type model: Any
    :param dummy_input: Dummy input data (Tensor, Tuple, or List of Dicts for detection)
    :type dummy_input: Union[torch.Tensor, tuple, list]
    :param model_fname: Optional path to model file for size estimation
    :type model_fname: Optional[str]
    :param runs: Number of timed runs
    :type runs: int
    :param warm_up_runs: Warm-up iterations before timing
    :type warm_up_runs: int
    :return: DataFrame with size, inference time, parameter count, etc.
    :rtype: pd.DataFrame
    """

    # Compute model size if applicable
    size_mb = os.path.getsize(model_fname) / 1024**2 if model_fname else None

    # Format input consistently
    if isinstance(dummy_input, (torch.Tensor, tuple)):
        dummy_tuple = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    else:
        dummy_tuple = dummy_input  # e.g., list of dicts for detection

    # Warm-up
    for _ in range(warm_up_runs):
        with torch.no_grad():
            if hasattr(model, "inference"):
                model.inference(*dummy_tuple)
            else:
                model(*dummy_tuple)

    # Measure inference time
    inference_times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            if hasattr(model, "inference"):
                model.inference(*dummy_tuple)
            else:
                model(*dummy_tuple)
        torch.cuda.synchronize()
        inference_times.append(time.time() - start)

    # Get number of parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Get input shape
    input_shape = get_data_shape(dummy_input)
    input_shape_str = "x".join(map(str, input_shape))

    result = {
        "input_shape": [input_shape_str],
        "n_params": [n_params],
        "size_mb": [size_mb],
        "inference_time_s": [np.mean(inference_times)],
    }

    return pd.DataFrame.from_dict(result)


class ImageDetectionTorchDataset(Dataset):
    """Dataset for image detection PyTorch models

    :param dataset: Image detection dataset
    :type dataset: ImageDetectionDataset
    :param transform: Transformation to be applied to images
    :type transform: transforms.Compose
    :param splits: Splits to be used from the dataset, defaults to ["test"]
    :type splits: str, optional
    """

    def __init__(
        self,
        dataset: dm_detection_dataset.ImageDetectionDataset,
        transform: transforms.Compose,
        splits: List[str] = ["test"],
    ):
        # Filter split and make filenames global
        dataset.dataset = dataset.dataset[dataset.dataset["split"].isin(splits)]
        self.dataset = dataset
        # Use the dataset's make_fname_global method instead of manual path joining
        self.dataset.make_fname_global()

        self.transform = transform

    def __len__(self):
        return len(self.dataset.dataset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load image and annotations, apply transforms.

        :param idx: Sample index
        :return: Tuple of (sample_id, image_tensor, target_dict)
        """
        row = self.dataset.dataset.iloc[idx]
        image_path = row["image"]
        ann_path = row["annotation"]

        image = Image.open(image_path).convert("RGB")
        boxes, labels, cat_ids = self.dataset.read_annotation(ann_path)

        # Convert boxes/labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # [N, 4]
        labels = torch.as_tensor(labels, dtype=torch.int64)  # [N]

        target = {
            "boxes": boxes,  # shape [N, 4] in [x1, y1, x2, y2] format
            "labels": labels,  # shape [N]
        }

        if self.transform:
            image, target = self.transform(image, target)

        return self.dataset.dataset.index[idx], image, target


class TorchImageDetectionModel(dm_detection_model.ImageDetectionModel):
    def __init__(
        self,
        model: Union[str, torch.nn.Module],
        model_cfg: str,
        ontology_fname: str,
        device: torch.device = None,
    ):
        """Image detection model for PyTorch framework

        :param model: Either the filename of a TorchScript model or the model already loaded into a PyTorch module.
        :type model: Union[str, torch.nn.Module]
        :param model_cfg: JSON file containing model configuration
        :type model_cfg: str
        :param ontology_fname: JSON file containing model output ontology
        :type ontology_fname: str
        :param device: torch.device to use (optional). If not provided, will auto-select cuda, mps, or cpu.
        """
        # Get device (GPU, MPS, or CPU) if not provided
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        # Load model from file or use passed instance
        if isinstance(model, str):
            assert os.path.isfile(model), "Torch model file not found"
            model_fname = model
            try:
                model = torch.jit.load(model, map_location=self.device)
                model_type = "compiled"
            except Exception:
                print(
                    "Model is not a TorchScript model. Loading as native PyTorch model."
                )
                model = torch.load(model, map_location=self.device)
                model_type = "native"
        elif isinstance(model, torch.nn.Module):
            model_fname = None
            model_type = "native"
        else:
            raise ValueError("Model must be a filename or a torch.nn.Module")

        # Init parent class
        super().__init__(model, model_type, model_cfg, ontology_fname, model_fname)
        self.model = self.model.to(self.device).eval()

        # --- Add reverse mapping for idx to class_name ---
        self.idx_to_class_name = {v["idx"]: k for k, v in self.ontology.items()}

        # Build input transforms (resize, normalize, etc.)
        self.transform_input = []

        if "resize" in self.model_cfg:
            self.transform_input += [
                transforms.Resize(
                    size=(
                        self.model_cfg["resize"].get("height", None),
                        self.model_cfg["resize"].get("width", None),
                    ),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ]

        if "crop" in self.model_cfg:
            crop_size = (
                self.model_cfg["crop"]["height"],
                self.model_cfg["crop"]["width"],
            )
            self.transform_input += [transforms.CenterCrop(crop_size)]

        try:
            self.transform_input += [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        except AttributeError:
            self.transform_input += [
                transforms.ToImageTensor(),
                transforms.ConvertDtype(torch.float32),
            ]

        if "normalization" in self.model_cfg:
            self.transform_input += [
                transforms.Normalize(
                    mean=self.model_cfg["normalization"]["mean"],
                    std=self.model_cfg["normalization"]["std"],
                )
            ]

        self.transform_input = transforms.Compose(self.transform_input)

    def inference(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Perform object detection inference for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: Dictionary with keys 'boxes', 'labels', 'scores'
        :rtype: Dict[str, torch.Tensor]
        """
        tensor = self.transform_input(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model(tensor)[0]  # Return only first image's result

        # Apply threshold filtering from model config
        confidence_threshold = self.model_cfg.get("confidence_threshold", 0.5)
        if confidence_threshold > 0:
            keep_mask = result["scores"] >= confidence_threshold
            result = {
                "boxes": result["boxes"][keep_mask],
                "labels": result["labels"][keep_mask],
                "scores": result["scores"][keep_mask],
            }

        return result

    def eval(
        self,
        dataset: dm_detection_dataset.ImageDetectionDataset,
        split: str | List[str] = "test",
        ontology_translation: Optional[str] = None,
        predictions_outdir: Optional[str] = None,
        results_per_sample: bool = False,
    ) -> pd.DataFrame:
        """Evaluate model over a detection dataset and compute metrics

        :param dataset: Image detection dataset
        :type dataset: ImageDetectionDataset
        :param split: Dataset split(s) to evaluate
        :type split: str | List[str]
        :param ontology_translation: Optional translation for class mapping
        :type ontology_translation: Optional[str]
        :param predictions_outdir: Directory to save predictions, if desired
        :type predictions_outdir: Optional[str]
        :param results_per_sample: Store per-sample metrics
        :type results_per_sample: bool
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        if results_per_sample and predictions_outdir is None:
            raise ValueError(
                "predictions_outdir required if results_per_sample is True"
            )

        if predictions_outdir is not None:
            os.makedirs(predictions_outdir, exist_ok=True)

        # Build LUT if ontology translation is provided
        lut_ontology = self.get_lut_ontology(dataset.ontology, ontology_translation)
        if lut_ontology is not None:
            lut_ontology = torch.tensor(lut_ontology, dtype=torch.int64).to(self.device)

        # Create DataLoader
        dataset = ImageDetectionTorchDataset(
            dataset,
            transform=self.transform_input,
            splits=[split] if isinstance(split, str) else split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.model_cfg.get("batch_size", 1),
            num_workers=self.model_cfg.get("num_workers", 1),
            collate_fn=lambda x: tuple(zip(*x)),  # handles variable-size targets
        )

        # Get iou_threshold from model config, default to 0.5 if not present
        iou_threshold = self.model_cfg.get("iou_threshold", 0.5)

        # Init metrics
        metrics_factory = um.DetectionMetricsFactory(
            iou_threshold=iou_threshold, num_classes=self.n_classes
        )

        with torch.no_grad():
            pbar = tqdm(dataloader, leave=True)
            for image_ids, images, targets in pbar:
                # Defensive check for empty images
                if not images or any(img.numel() == 0 for img in images):
                    print("Skipping batch: empty image tensor detected.")
                    continue

                images = [img.to(self.device) for img in images]
                predictions = self.model(images)

                for i in range(len(images)):
                    gt = targets[i]
                    pred = predictions[i]

                    # Apply confidence threshold filtering
                    confidence_threshold = self.model_cfg.get(
                        "confidence_threshold", 0.5
                    )
                    if confidence_threshold > 0:
                        keep_mask = pred["scores"] >= confidence_threshold
                        pred = {
                            "boxes": pred["boxes"][keep_mask],
                            "labels": pred["labels"][keep_mask],
                            "scores": pred["scores"][keep_mask],
                        }

                    # Apply ontology translation if needed
                    if lut_ontology is not None:
                        gt["labels"] = lut_ontology[gt["labels"]]

                    # Update metrics
                    metrics_factory.update(
                        gt["boxes"],
                        gt["labels"],
                        pred["boxes"],
                        pred["labels"],
                        pred["scores"],
                    )

                    # Store predictions if needed
                    if predictions_outdir is not None:
                        sample_id = image_ids[i]
                        pred_boxes = pred["boxes"].cpu().numpy()
                        pred_labels = pred["labels"].cpu().numpy()
                        pred_scores = pred["scores"].cpu().numpy()
                        out_data = []

                        for box, label, score in zip(
                            pred_boxes, pred_labels, pred_scores
                        ):
                            # Convert label index to class name using model ontology
                            class_name = self.idx_to_class_name.get(
                                int(label), f"class_{label}"
                            )
                            out_data.append(
                                {
                                    "image_id": sample_id,
                                    "label": class_name,
                                    "score": float(score),
                                    "bbox": box.tolist(),
                                }
                            )

                        df = pd.DataFrame(out_data)
                        df.to_json(
                            os.path.join(predictions_outdir, f"{sample_id}.json"),
                            orient="records",
                            indent=2,
                        )

                        if results_per_sample:
                            sample_mf = um.DetectionMetricsFactory(
                                iou_threshold=iou_threshold, num_classes=self.n_classes
                            )
                            sample_mf.update(
                                gt["boxes"],
                                gt["labels"],
                                pred["boxes"],
                                pred["labels"],
                                pred["scores"],
                            )
                            sample_df = sample_mf.get_metrics_dataframe(self.ontology)
                            sample_df.to_csv(
                                os.path.join(
                                    predictions_outdir, f"{sample_id}_metrics.csv"
                                )
                            )

        return metrics_factory.get_metrics_dataframe(self.ontology)

    def get_computational_cost(
        self, image_size: Tuple[int], runs: int = 30, warm_up_runs: int = 5
    ) -> dict:
        """Get computational cost metrics like inference time

        :param image_size: Size of input image (H, W)
        :type image_size: Tuple[int]
        :param runs: Number of repeated runs to average over
        :type runs: int
        :param warm_up_runs: Warm-up runs before timing
        :type warm_up_runs: int
        :return: Dictionary with computational cost details
        :rtype: dict
        """
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
        return get_computational_cost(
            self.model, dummy_input, self.model_fname, runs, warm_up_runs
        )
