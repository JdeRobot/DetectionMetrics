from collections import OrderedDict
import logging
import os

import pandas as pd

from detectionmetrics.datasets.dataset import ImageSegmentationDataset
import detectionmetrics.utils.io as uio


class Rellis3dImageSegmentationDataset(ImageSegmentationDataset):
    """Specific class for Rellis3D-styled image segmentation datasets. All data can
    be downloaded from the official repo (https://github.com/unmannedlab/RELLIS-3D):
        images   -> https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV
        labels   -> https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5
        split    -> https://drive.google.com/file/d/1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ
        ontology -> https://drive.google.com/file/d/1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6

    :param dataset_dir: Directory where both RGB images and annotations have been
    extracted to
    :type dataset_dir: str
    :param split_dir: Directory where train, val, and test files (.lst) have been
    extracted to
    :type split_dir: str
    :param ontology_fname: YAML file contained in the ontology compressed directory
    :type ontology_fname: str
    """

    def __init__(self, dataset_dir: str, split_dir: str, ontology_fname: str):
        super().__init__()

        # Check that provided paths exist
        assert os.path.isdir(dataset_dir), "Dataset directory not found"
        assert os.path.isdir(split_dir), "Split directory not found"
        assert os.path.isfile(ontology_fname), "Ontology file not found"

        # Load and adapt ontology
        names, colors = uio.read_yaml(ontology_fname)
        self.ontology = {names[idx]: {"idx": idx, "rgb": colors[idx]} for idx in names}

        # Get samples filenames
        train_split = uio.read_txt(os.path.join(split_dir, "train.lst"))
        val_split = uio.read_txt(os.path.join(split_dir, "val.lst"))
        test_split = uio.read_txt(os.path.join(split_dir, "test.lst"))

        train_split = [s.split(" ") + ["train"] for s in train_split]
        val_split = [s.split(" ") + ["val"] for s in val_split]
        test_split = [s.split(" ") + ["test"] for s in test_split]

        samples_data = train_split + val_split + test_split

        # Build dataset as ordered python dictionary
        dataset = OrderedDict()
        skipped_samples = []
        for image_fname, label_fname, split in samples_data:
            sample_dir, sample_base_name = os.path.split(image_fname)
            sample_base_name, _ = os.path.splitext(sample_base_name)
            scene = os.path.split(os.path.split(sample_dir)[0])[-1]
            sample_name = f"{scene}-{sample_base_name}"

            image_fname = os.path.join(dataset_dir, image_fname)
            label_fname = os.path.join(dataset_dir, label_fname)

            if not os.path.isfile(image_fname) or not os.path.isfile(label_fname):
                missing_file = "image" if not os.path.isfile(label_fname) else "label"
                logging.warning(f"Missing {missing_file} for {sample_name}. Skipped!")
                skipped_samples.append(sample_name)
                continue

            dataset[sample_name] = (image_fname, label_fname, scene, split)

        # Convert to Pandas
        cols = ["image", "label", "scene", "split"]
        self.dataset = pd.DataFrame.from_dict(dataset, orient="index", columns=cols)
        self.dataset.attrs = {"ontology": self.ontology}

        # Report results
        print(f"Samples retrieved: {len(dataset)} / {len(samples_data)}")
        if skipped_samples:
            print("Skipped samples:")
            for sample_name in skipped_samples:
                print(f"\n\t{sample_name}")
