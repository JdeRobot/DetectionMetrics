from unittest.mock import patch

import numpy as np
import pandas as pd

from tabs.tasks.lidar_segmentation import dataset_viewer


def test_lidar_dataset_cache_and_subsampling() -> None:
    """Regression tests for LiDAR GUI dataset caching and point subsampling."""
    session_state = {}

    with patch.object(dataset_viewer.st, "session_state", session_state):
        with patch.object(
            dataset_viewer, "SemanticKITTILiDARSegmentationDataset"
        ) as dataset_cls:
            first_dataset = dataset_viewer.load_semantic_kitti_dataset(
                "/fake/semantic-kitti",
                "/fake/semantic-kitti.yaml",
                "val",
            )
            second_dataset = dataset_viewer.load_semantic_kitti_dataset(
                "/fake/semantic-kitti",
                "/fake/semantic-kitti.yaml",
                "val",
            )

    dataset_cls.assert_called_once_with(
        dataset_dir="/fake/semantic-kitti",
        config_fname="/fake/semantic-kitti.yaml",
        split="val",
    )
    assert first_dataset is second_dataset

    points = np.arange(20, dtype=np.float32).reshape(5, 4)
    labels = np.array([10, 11, 12, 13, 14], dtype=np.uint32)
    sampled_points, sampled_labels = dataset_viewer.subsample_points(
        points,
        labels,
        max_points=3,
    )

    assert np.array_equal(sampled_points, points[[0, 2, 4]])
    assert np.array_equal(sampled_labels, labels[[0, 2, 4]])


def test_lidar_visualization_helpers() -> None:
    """Regression tests for LiDAR GUI colors, hover names, and class table."""
    labels = np.array([0, 1, 99], dtype=np.uint32)
    ontology = {
        "unlabeled": {"idx": 0, "rgb": [0, 0, 0]},
        "car": {"idx": 1, "rgb": [255, 0, 0]},
    }

    label_colors = dataset_viewer.get_label_colors(labels, ontology)
    label_names = dataset_viewer.get_label_names(labels, ontology)
    intensity_colors = dataset_viewer.intensity_to_colors(
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
        clip_range=(0.25, 0.75),
    )
    classes = dataset_viewer.classes_dataframe(ontology)

    expected_label_colors = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    expected_intensity_colors = np.array(
        [
            [0.0, 0.35, 0.05],
            [0.5, 0.65, 0.025],
            [1.0, 0.95, 0.0],
        ],
        dtype=np.float32,
    )
    expected_classes = pd.DataFrame(
        [
            {"class": "unlabeled", "id": 0, "rgb": [0, 0, 0]},
            {"class": "car", "id": 1, "rgb": [255, 0, 0]},
        ]
    )

    assert np.array_equal(label_colors, expected_label_colors)
    assert label_names == ["unlabeled", "car", "99"]
    assert np.allclose(intensity_colors, expected_intensity_colors)
    pd.testing.assert_frame_equal(classes, expected_classes)
