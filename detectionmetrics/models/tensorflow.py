import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tqdm import tqdm

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
from detectionmetrics.models.model import ImageSegmentationModel
import detectionmetrics.utils.conversion as uc
import detectionmetrics.utils.io as uio
import detectionmetrics.utils.metrics as um


class ImageSegmentationTensorflowDataset:
    """Dataset for image segmentation Tensorflow models

    :param dataset: Image segmentation dataset
    :type dataset: ImageSegmentationDataset
    :param image_size: Image size in pixels (width, height)
    :type image_size: Tuple[int, int]
    :param batch_size: Batch size, defaults to 1
    :type batch_size: int, optional
    :param split: Split to be used from the dataset, defaults to "all"
    :type split: str, optional
    """

    def __init__(
        self,
        dataset: ImageSegmentationDataset,
        image_size: Tuple[int, int],
        batch_size: int = 1,
        split: str = "all",
    ):
        self.image_size = image_size

        # Filter split and make filenames global
        if split != "all":
            dataset.dataset = dataset.dataset[dataset.dataset["split"] == split]
        dataset.make_fname_global()

        # Build tensorflow dataset
        fnames = (dataset.dataset["image"], dataset.dataset["label"])
        self.dataset = tf_data.Dataset.from_tensor_slices(fnames)
        self.dataset = self.dataset.map(
            self.load_data, num_parallel_calls=tf_data.AUTOTUNE
        )
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

    def read_image(self, fname: str, label=False) -> tf.Tensor:
        """Read a single image or label

        :param fname: Input image or label filename
        :type fname: str
        :param label: Whether the input data is a label or not, defaults to False
        :type label: bool, optional
        :return: Tensorflow tensor containing read image or label
        :rtype: tf.Tensor
        """
        # Use nearest neighbors to avoid interpolation when dealing with labels
        resize_method = "nearest" if label else "bilinear"
        n_channels = 1 if label else 3

        image = tf_io.read_file(fname)
        image = tf_image.decode_png(image, channels=n_channels)
        image.set_shape([None, None, n_channels])
        image = tf_image.resize(
            images=image, size=self.image_size, method=resize_method
        )

        return image

    def load_data(
        self, images_fnames: List[str], labels_fnames: List[str]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Function for loading data for each dataset sample

        :param images_fnames: List containing all image filenames
        :type images_fnames: List[str]
        :param labels_fnames: List containing all corresponding label filenames
        :type labels_fnames: List[str]
        :return: Image and label tensor pairs
        :rtype: Tuple[tf.Tensor, tf.Tensor]
        """
        image = self.read_image(images_fnames)
        label = self.read_image(labels_fnames, label=True)
        return image, label


class TensorflowImageSegmentationModel(ImageSegmentationModel):
    """Image segmentation model for Tensorflow framework

    :param model_fname: Tensorflow model in SavedModel format
    :type model_fname: str
    :param model_cfg: JSON file containing model configuration
    :type model_cfg: str
    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    """

    def __init__(self, model_fname: str, model_cfg: str, ontology_fname: str):
        super().__init__(ontology_fname, model_cfg)

        # Check that provided path exist and load model
        assert os.path.isdir(model_fname), "Model file not found"
        self.model = tf.saved_model.load(model_fname)

        # Init transformation for input images
        def t_in(image):
            tensor = tf.convert_to_tensor(image)
            tensor = tf_image.resize(images=tensor, size=self.model_cfg["image_size"])
            return tensor

        self.t_in = t_in

        # Init transformation for model output
        self.t_out = lambda x: Image.fromarray(
            tf.argmax(tf.squeeze(x), axis=2).numpy().astype(np.uint8)
        )

    def inference(self, image: Image.Image) -> Image.Image:
        """Perform inference for a single image

        :param image: PIL image
        :type image: Image.Image
        :return: segmenation result as PIL image
        :rtype: Image.Image
        """
        tensor = self.t_in(image)

        # TODO: check if this is consistent across different models
        result = self.model.signatures["serving_default"](tensor)
        if isinstance(result, dict):
            result = result["output_0"]

        return self.t_out(result)

    def eval(
        self,
        dataset: ImageSegmentationDataset,
        batch_size: int = 1,
        split: str = "all",
        ontology_translation: Optional[str] = None,
    ) -> pd.DataFrame:
        """Perform evaluation for an image segmentation dataset

        :param dataset_test: Image segmentation dataset for which the evaluation will
        be performed
        :type dataset_test: ImageSegmentationDataset
        :param batch_size: Batch size, defaults to 1
        :type batch_size: int, optional
        :param split: Split to be used from the dataset, defaults to "all"
        :type split: str, optional
        :param ontology_translation: JSON file containing translation between dataset
        and model output ontologies
        :type ontology_translation: str, optional
        :return: DataFrame containing evaluation results
        :rtype: pd.DataFrame
        """
        # Build a LUT for transforming ontology if needed
        lut_ontology = None
        if dataset.ontology != self.ontology:
            if ontology_translation is not None:
                ontology_translation = uio.read_json(ontology_translation)
            lut_ontology = uc.get_ontology_conversion_lut(
                dataset.ontology, self.ontology, ontology_translation
            )
            keys = tf.constant(range(len(lut_ontology)), dtype=tf.int32)
            values = tf.constant(list(lut_ontology), dtype=tf.int32)
            lut_ontology = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
                default_value=0,
            )

        # Get Tensorflow dataset
        dataset = ImageSegmentationTensorflowDataset(
            dataset,
            image_size=self.model_cfg["image_size"],
            batch_size=batch_size,
            split=split,
        )

        # Init metrics
        results = {}
        iou = um.MeanIoU(self.n_classes)

        # Evaluation loop
        pbar = tqdm(dataset.dataset)
        for image, label in pbar:
            pred = self.model.signatures["serving_default"](image)
            if isinstance(pred, dict):
                pred = pred["output_0"]

            label = tf.squeeze(label, axis=3)
            if lut_ontology is not None:
                label = tf.cast(lut_ontology.lookup(tf.cast(label, tf.int32)), tf.uint8)

            pred = tf.argmax(pred, axis=3)
            pred = tf.one_hot(pred, self.n_classes)
            pred = tf.transpose(pred, perm=[0, 3, 1, 2]).numpy()

            label = tf.one_hot(label, self.n_classes)
            label = tf.transpose(label, perm=[0, 3, 1, 2]).numpy()

            iou.update(pred, label)

        # Get metrics results
        iou = [float(n) for n in iou.compute()]

        # Build results dataframe
        results = {}
        for class_name, class_data in self.ontology.items():
            results[class_name] = {"iou": iou[class_data["idx"]]}
        results["global"] = {"iou": np.nanmean(iou)}

        return pd.DataFrame(results)
