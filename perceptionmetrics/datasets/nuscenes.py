import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes

from perceptionmetrics.datasets.detection import LiDARDetectionDataset


def quaternion_yaw(rotation) -> float:
    """Return yaw angle from a nuScenes quaternion.

    nuScenes stores rotations as quaternions in ``[w, x, y, z]`` order.
    """
    w, x, y, z = rotation
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def build_nuscenes_lidar_detection_dataset(
    dataset_dir: str,
    version: str = "v1.0-mini",
    split: str = "train",
    nusc_object: Optional[NuScenes] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Build a nuScenes LiDAR detection dataset index.

    Each row stores the LiDAR point cloud path and the corresponding
    ``sample_data`` token used to retrieve 3D annotations.

    :param dataset_dir: Path to the nuScenes dataset root directory
    :type dataset_dir: str
    :param version: nuScenes dataset version, defaults to "v1.0-mini"
    :type version: str
    :param split: Dataset split to assign to the indexed samples
    :type split: str
    :param nusc_object: Optional pre-initialized NuScenes object
    :type nusc_object: Optional[NuScenes]
    :return: Dataset DataFrame and ontology dictionary
    :rtype: Tuple[pd.DataFrame, dict]
    """
    dataset_dir = os.path.abspath(dataset_dir)
    assert os.path.isdir(
        dataset_dir
    ), f"Dataset directory {dataset_dir} does not exist."

    nusc = (
        nusc_object
        if nusc_object
        else NuScenes(version=version, dataroot=dataset_dir, verbose=False)
    )

    ontology = {
        category["name"]: {"idx": idx + 1, "rgb": [0, 0, 0]}
        for idx, category in enumerate(nusc.category)
    }

    rows = []
    sample_names = []
    for sample in nusc.sample:
        lidar_token = sample["data"].get("LIDAR_TOP")
        if lidar_token is None:
            continue

        sample_data = nusc.get("sample_data", lidar_token)
        rows.append(
            {
                "points": os.path.join(dataset_dir, sample_data["filename"]),
                "annotation": lidar_token,
                "split": split,
            }
        )
        sample_names.append(sample["token"])

    dataset = pd.DataFrame(rows, index=sample_names)
    dataset.attrs = {"ontology": ontology}

    return dataset, ontology


class NuScenesLiDARDetectionDataset(LiDARDetectionDataset):
    """Dataset class for nuScenes LiDAR 3D object detection.

    The annotation boxes are returned as ``[x, y, z, dx, dy, dz, yaw]`` in the
    LiDAR/global annotation convention used by nuScenes metadata.

    :param dataset_dir: Path to the nuScenes dataset root directory
    :type dataset_dir: str
    :param version: nuScenes dataset version, defaults to "v1.0-mini"
    :type version: str
    :param split: Dataset split to assign to the indexed samples
    :type split: str
    """

    def __init__(
        self,
        dataset_dir: str,
        version: str = "v1.0-mini",
        split: str = "train",
    ):
        dataset_dir = os.path.abspath(dataset_dir)
        assert os.path.isdir(
            dataset_dir
        ), f"Dataset directory {dataset_dir} does not exist."

        self.nusc = NuScenes(version=version, dataroot=dataset_dir, verbose=False)
        dataset, ontology = build_nuscenes_lidar_detection_dataset(
            dataset_dir=dataset_dir,
            version=version,
            split=split,
            nusc_object=self.nusc,
        )
        self.cat_to_idx = {
            category_name: class_data["idx"]
            for category_name, class_data in ontology.items()
        }

        super().__init__(
            dataset=dataset,
            dataset_dir=dataset_dir,
            ontology=ontology,
            is_kitti_format=False,
        )

    def make_fname_global(self):
        """Make point cloud paths absolute while keeping annotation tokens intact."""
        if self.dataset_dir is not None:
            self.dataset["points"] = self.dataset["points"].apply(
                lambda x: os.path.join(self.dataset_dir, x)
                if x is not None and not os.path.isabs(x)
                else x
            )
            self.dataset_dir = None

    def read_points(self, fname: str) -> np.ndarray:
        """Read nuScenes LiDAR points.

        nuScenes LiDAR ``.bin`` files store five float32 values per point:
        ``x, y, z, intensity, ring_index``.
        """
        return np.fromfile(fname, dtype=np.float32).reshape(-1, 5)

    def read_annotation(self, fname: str):
        """Read 3D annotations for a nuScenes LiDAR sample-data token.

        :param fname: nuScenes ``sample_data`` token
        :type fname: str
        :return: Boxes in ``[x, y, z, dx, dy, dz, yaw]`` format and class IDs
        :rtype: Tuple[list, list]
        """
        sample_data = self.nusc.get("sample_data", fname)
        sample = self.nusc.get("sample", sample_data["sample_token"])

        boxes = []
        labels = []
        for annotation_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", annotation_token)
            category_name = annotation["category_name"]
            if category_name not in self.cat_to_idx:
                continue

            box = [
                *annotation["translation"],
                *annotation["size"],
                quaternion_yaw(annotation["rotation"]),
            ]
            boxes.append(box)
            labels.append(self.cat_to_idx[category_name])

        return boxes, labels
