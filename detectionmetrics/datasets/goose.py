from collections import OrderedDict
from glob import glob
import os
from typing import Optional

import pandas as pd

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
import detectionmetrics.utils.conversion as uc


class GooseImageSegmentationDataset(ImageSegmentationDataset):
    """Specific class for GOOSE-styled image segmentation datasets. All data can be
    downloaded from the official webpage (https://goose-dataset.de):
        train -> https://goose-dataset.de/storage/goose_2d_train.zip
        val   -> https://goose-dataset.de/storage/goose_2d_val.zip
        test  -> https://goose-dataset.de/storage/goose_2d_test.zip

    :param train_dataset_dir: Directory containing training data
    :type train_dataset_dir: str
    :param val_dataset_dir: Directory containing validation data, defaults to None
    :type val_dataset_dir: str, optional
    :param test_dataset_dir: Directory containing test data, defaults to None
    :type test_dataset_dir: str, optional
    """

    def __init__(
        self,
        train_dataset_dir: str,
        val_dataset_dir: Optional[str] = None,
        test_dataset_dir: Optional[str] = None,
    ):
        super().__init__()

        # Check that provided paths exist
        assert os.path.isdir(train_dataset_dir), "Train dataset directory not found"
        dataset_dirs = {"train": train_dataset_dir}
        if val_dataset_dir is not None:
            assert os.path.isdir(val_dataset_dir)
            dataset_dirs["val"] = val_dataset_dir, "Val. dataset directory not found"
        if test_dataset_dir is not None:
            assert os.path.isdir(test_dataset_dir)
            dataset_dirs["test"] = test_dataset_dir, "Test dataset directory not found"

        # Load and adapt ontology
        ontology_fname = os.path.join(train_dataset_dir, "goose_label_mapping.csv")
        assert os.path.isfile(ontology_fname), "Ontology file not found"

        ontology = pd.read_csv(ontology_fname)
        self.ontology = {}
        for idx, (name, _, _, color) in ontology.iterrows():
            self.ontology[name] = {"idx": idx, "rgb": uc.hex_to_rgb(color)}

        # Build dataset as ordered python dictionary
        dataset = OrderedDict()
        for split, dataset_dir in dataset_dirs.items():
            train_images = os.path.join(dataset_dir, f"images/{split}/*/*_vis.png")
            for image_fname in glob(train_images):
                sample_dir, sample_base_name = os.path.split(image_fname)
                sample_base_name, _ = os.path.splitext(sample_base_name.split("__")[-1])
                sample_base_name = sample_base_name.split("_windshield")[0]

                scene = os.path.split(sample_dir)[-1]
                sample_name = f"{scene}-{sample_base_name}"

                label_base_name = f"{scene}__{sample_base_name}_labelids.png"
                label_fname = os.path.join(
                    dataset_dir, "labels", scene, label_base_name
                )
                label_fname = None if not os.path.isfile(label_fname) else label_fname

                image_fname = os.path.join(dataset_dir, image_fname)
                dataset[sample_name] = (image_fname, label_fname, scene, split)

        # Convert to Pandas
        cols = ["image", "label", "scene", "split"]
        self.dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)

        # Report results
        print(f"Samples retrieved: {len(dataset)}")
