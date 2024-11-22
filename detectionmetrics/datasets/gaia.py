import os
import pandas as pd

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
import detectionmetrics.utils.io as uio


class GaiaImageSegmentationDataset(ImageSegmentationDataset):
    """Specific class for GAIA-styled image segmentation datasets

    :param dataset_fname: Parquet dataset filename
    :type dataset_fname: str
    """

    def __init__(self, dataset_fname: str):
        super().__init__()

        # Check that provided path exist
        assert os.path.isfile(dataset_fname), "Dataset file not found"

        # Load dataset Parquet file
        self.dataset = pd.read_parquet(dataset_fname)
        self.dataset_dir = os.path.dirname(dataset_fname)

        # Read ontology file
        ontology_fname = self.dataset.attrs["ontology_fname"]
        self.ontology = uio.read_json(os.path.join(self.dataset_dir, ontology_fname))

        # Report results
        print(f"Samples retrieved: {self.dataset}")
