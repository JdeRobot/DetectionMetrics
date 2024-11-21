from abc import ABC, abstractmethod
import os
from typing import Optional

import pandas as pd
from PIL import Image

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
import detectionmetrics.utils.io as uio


class ImageSegmentationModel(ABC):
    """Parent image segmentation model class

    :param ontology_fname: JSON file containing model output ontology
    :type ontology_fname: str
    :param model_cfg: JSON file containing model configuration (e.g. image size or
    normalization parameters)
    :type model_cfg: str
    """

    def __init__(self, ontology_fname: str, model_cfg: str):
        # Check that provided paths exist
        assert os.path.isfile(ontology_fname), "Ontology file not found"
        assert os.path.isfile(model_cfg), "Model configuration not found"

        # Read ontology and model configuration
        self.ontology = uio.read_json(ontology_fname)
        self.model_cfg = uio.read_json(model_cfg)
        self.n_classes = len(self.ontology)

        self.model = None

    @abstractmethod
    def inference(self, image: Image.Image) -> Image.Image:
        """Perform inference for a single image

        :param image: PIL image.
        :type image: Image.Image
        :return: segmenation result as PIL image
        :rtype: Image.Image
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
