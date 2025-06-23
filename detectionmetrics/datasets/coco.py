from pycocotools.coco import COCO
import os
import pandas as pd
from typing import Tuple,List 

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

    def read_annotation(self, fname: str) -> Tuple[List[List[float]], List[int]]:
        """Return bounding boxes and labels for a given image ID.

        :param fname: str (image_id in string form)
        :return: Tuple of (boxes, labels)
        """
        # Extract image ID (fname might be a path or ID string)
        try:
            image_id = int(os.path.basename(fname))  # handles both '123' and '/path/to/123'
        except ValueError:
            raise ValueError(f"Invalid annotation ID: {fname}")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        for ann in anns:
            # Convert [x, y, width, height] to [x1, y1, x2, y2]
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        return boxes, labels

