from typing import List, Optional, Tuple

from mmdet3d.datasets.transforms import (
    LoadPointsFromFile,
    LoadAnnotations3D,
    Pack3DDetInputs,
)
from mmengine.registry import FUNCTIONS
import torch
from torchvision.transforms import Compose

COLLATE_FN = FUNCTIONS.get("pseudo_collate")


def get_sample(
    points_fname: str,
    model_cfg: dict,
    label_fname: Optional[str] = None,
    name: Optional[str] = None,
    idx: Optional[int] = None,
) -> dict:
    """Get sample data for mmdetection3d models

    :param points_fname: filename of the point cloud
    :type points_fname: str
    :param model_cfg: model configuration
    :type model_cfg: dict
    :param label_fname: filename of the semantic label, defaults to None
    :type label_fname: Optional[str], optional
    :param name: sample name, defaults to None
    :type name: Optional[str], optional
    :param idx: sample numerical index, defaults to None
    :type idx: Optional[int], optional
    :return: Sample data dictionary
    :rtype: dict
    """
    sample = {
        "lidar_points": {
            "lidar_path": points_fname,
            "num_pts_feats": model_cfg.get("n_feats", 4),
        },
        "pts_semantic_mask_path": label_fname,
        "sample_id": name,
        "sample_idx": idx,
        "num_pts_feats": model_cfg.get("n_feats", 4),
        "lidar_path": points_fname,
    }

    n_feats = sample["num_pts_feats"]
    transforms = [LoadPointsFromFile(coord_type="LIDAR", load_dim=4, use_dim=n_feats)]
    if sample["pts_semantic_mask_path"] is not None:
        transforms.append(
            LoadAnnotations3D(
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype="np.uint32",
                seg_offset=65536,
                dataset_type="semantickitti",
            )
        )
    transforms.append(
        Pack3DDetInputs(
            keys=["points", "pts_semantic_mask"],
            meta_keys=["sample_idx", "lidar_path", "num_pts_feats", "sample_id"],
        )
    )
    transforms = Compose(transforms)

    return transforms(sample)


def inference(
    sample: dict, model: torch.nn.Module, model_cfg: dict
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """Perform inference on a sample using an mmdetection3D model

    :param sample: sample data dictionary
    :type sample: dict
    :param model: mmdetection3D model
    :type model: torch.nn.Module
    :param model_cfg: model configuration
    :type model_cfg: dict
    :return: predictions, labels, and sample names
    :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]
    """
    single_sample = not isinstance(sample["data_samples"], list)
    if single_sample:
        sample = COLLATE_FN([sample])

    sample = model.data_preprocessor(sample, training=False)
    inputs, data_samples = sample["inputs"], sample["data_samples"]
    has_labels = hasattr(data_samples[0].gt_pts_seg, "pts_semantic_mask")

    outputs = model(inputs, data_samples, mode="predict")

    preds, labels, names = ([], [], []) if has_labels else ([], None, None)
    for output in outputs:
        preds.append(output.pred_pts_seg.pts_semantic_mask)
        if has_labels:
            labels.append(output.gt_pts_seg.pts_semantic_mask)
            names.append(output.metainfo["sample_id"])
    preds = torch.stack(preds, dim=0).squeeze()
    if has_labels:
        labels = torch.stack(labels, dim=0).squeeze()

    return preds, labels, names
