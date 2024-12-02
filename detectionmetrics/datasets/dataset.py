import os
import shutil
from typing_extensions import Self

import cv2
import pandas as pd
from tqdm import tqdm

import detectionmetrics.utils.io as uio


class ImageSegmentationDataset:
    """Parent image segmentation dataset class
    """

    def __init__(self):
        self.dataset = pd.DataFrame([])
        self.dataset_dir = None
        self.ontology = {}

    def __len__(self):
        return len(self.dataset)

    def append(self, new_dataset: Self):
        """Append another dataset with common ontology

        :param new_dataset: Dataset to be appended
        :type new_dataset: Self
        """
        assert self.ontology == new_dataset.ontology, "Ontologies don't match"

        # Global filenames to avoid dealing with each dataset relative location
        self.make_fname_global()
        new_dataset.make_fname_global()

        # Simply concatenate pandas dataframes
        self.dataset = pd.concat(
            [self.dataset, new_dataset.dataset], verify_integrity=True
        )

    def make_fname_global(self):
        """Get all relative filenames in dataset and make global
        """
        if self.dataset_dir is not None:
            self.dataset["image"] = self.dataset["image"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset["label"] = self.dataset["label"].apply(
                lambda x: os.path.join(self.dataset_dir, x) if x is not None else None
            )
            self.dataset_dir = None  # dataset_dir=None -> filenames must be relative

    def export(self, outdir: str):
        """Export dataset dataframe and image files

        :param outdir: Directory where Parquet and image files will be stored
        :type outdir: str
        """
        os.makedirs(outdir, exist_ok=True)

        pbar = tqdm(self.dataset.iterrows())

        for sample_name, row in pbar:
            pbar.set_description(f"Exporting sample: {sample_name}")

            # Create each split directory
            split = row["split"]
            split_dir = os.path.join(outdir, split)
            if not os.path.isdir(split_dir):
                os.makedirs(split_dir, exist_ok=True)

            # Init target filenames for both images and labels
            rel_image_fname = os.path.join(split, f"image-{sample_name}.png")
            rel_label_fname = os.path.join(split, f"label-{sample_name}.png")

            image_fname = row["image"]
            label_fname = row["label"]
            if self.dataset_dir is not None:
                image_fname = os.path.join(self.dataset_dir, image_fname)
                if label_fname:
                    label_fname = os.path.join(self.dataset_dir, label_fname)

            # If image mode is not appropriate: read, convert, and rewrite image
            if uio.get_image_mode(image_fname) != "RGB":
                image = cv2.imread(image_fname, 1)  # convert to RGB
                cv2.imwrite(os.path.join(outdir, rel_image_fname), image)
            # if image mode is appropriate simply copy image to new location
            else:
                shutil.copy2(image_fname, os.path.join(outdir, rel_image_fname))
            self.dataset.at[sample_name, "image"] = rel_image_fname

            # Same for labels
            if label_fname:
                if uio.get_image_mode(label_fname) != "L":
                    label = cv2.imread(label_fname, 0)  # convert to L
                    cv2.imwrite(os.path.join(outdir, rel_label_fname), label)
                else:
                    shutil.copy2(label_fname, os.path.join(outdir, rel_label_fname))
                self.dataset.at[sample_name, "label"] = rel_label_fname

        self.dataset_dir = outdir

        # Write ontology and store relative path in dataset attributes
        ontology_fname = "ontology.json"
        self.dataset.attrs = {"ontology_fname": ontology_fname}
        uio.write_json(os.path.join(outdir, ontology_fname), self.ontology)

        # Store dataset as Parquet file containing relative filenames
        self.dataset.to_parquet(os.path.join(outdir, "dataset.parquet"))
