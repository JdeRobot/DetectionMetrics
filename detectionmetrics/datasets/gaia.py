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
        # Check that provided path exist
        assert os.path.isfile(dataset_fname), "Dataset file not found"

        # Load dataset Parquet file
        dataset = pd.read_parquet(dataset_fname)
        dataset_dir = os.path.dirname(dataset_fname)

        # Read ontology file
        ontology_fname = dataset.attrs["ontology_fname"]
        ontology = uio.read_json(os.path.join(dataset_dir, ontology_fname))
        for name, data in ontology.items():
            ontology[name]["rgb"] = tuple(data["rgb"])

        # Report results
        print(f"Samples retrieved: {len(dataset)}")

        super().__init__(dataset, dataset_dir, ontology)
