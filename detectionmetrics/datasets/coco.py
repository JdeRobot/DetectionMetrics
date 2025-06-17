from pycocotools.coco import COCO
import os
import pandas as pd

from detectionmetrics.datasets.detection import ImageDetectionDataset


class CocoDataset(ImageDetectionDataset):
    """
    Specific class for COCO-styled object detection datasets.

    :param annotation_file: Path to the COCO-format JSON annotation file
    :type annotation_file: str
    :param image_dir: Path to the directory containing image files
    :type image_dir: str
    """
    def __init__(self, annotation_file: str, image_dir: str):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.ontology = self._build_ontology()
        self.dataset = self._build_dataframe()

        super().__init__(dataset=self.dataset, dataset_dir=image_dir, ontology=self.ontology)

    def _build_ontology(self):
        """Build ontology dict from COCO categories."""
        ontology = {}
        for cat in self.coco.loadCats(self.coco.getCatIds()):
            ontology[cat["name"]] = {
                "idx": cat["id"],
                "name": cat["name"],
                "rgb": [0, 0, 0]  # Placeholder; COCO doesn't define RGB colors
            }
        return ontology

    def _build_dataframe(self):
        """Build dataset DataFrame from COCO image IDs."""
        rows = []
        for img_id in self.coco.getImgIds():
            img_info = self.coco.loadImgs(img_id)[0]
            rows.append({
                "image": img_info["file_name"],
                "annotation": str(img_id),
                "split": "train"  # Update if needed
            })
        return pd.DataFrame(rows)

    def read_annotation(self, fname: str):
        """Read and return COCO-format annotations.

        :param fname: COCO annotation file path.
        :return: List of annotation dictionaries.
        """
        image_id = int(fname)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        return [
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "iscrowd": ann.get("iscrowd", 0),
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "id": ann.get("id")
            }
            for ann in anns
        ]
