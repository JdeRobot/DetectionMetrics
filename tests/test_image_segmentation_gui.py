from unittest.mock import patch

import numpy as np
from PIL import Image
import pandas as pd

from tabs.tasks.image_segmentation import dataset_viewer


def test_image_segmentation_dataset_loader() -> None:
    """Regression tests for image segmentation GUI dataset loading.

    Verifies that:
    - Cityscapes sidebar values are passed into the dataset class;
    - repeated loads reuse the same session-state cache entry.
    """
    session_state = {
        "segmentation_image_dir": "images",
        "segmentation_label_dir": "labels",
        "segmentation_image_suffix": "_image.png",
        "segmentation_label_suffix": "_label.png",
        "segmentation_use_train_id": True,
    }

    with patch.object(dataset_viewer.st, "session_state", session_state):
        with patch.object(
            dataset_viewer, "CityscapesImageSegmentationDataset"
        ) as dataset_cls:
            first_dataset = dataset_viewer.load_cityscapes_dataset(
                "/fake/cityscapes", "val"
            )
            second_dataset = dataset_viewer.load_cityscapes_dataset(
                "/fake/cityscapes", "val"
            )

    dataset_cls.assert_called_once_with(
        train_dataset_root=None,
        val_dataset_root="/fake/cityscapes",
        test_dataset_root=None,
        image_dir="images",
        label_dir="labels",
        image_suffix="_image.png",
        label_suffix="_label.png",
        use_train_id=True,
    )
    assert first_dataset is second_dataset


def test_image_segmentation_visualization_helpers() -> None:
    """Regression tests for mask overlay and classes table helpers."""
    image = Image.fromarray(
        np.array(
            [
                [[10, 10, 10], [20, 20, 20]],
                [[30, 30, 30], [40, 40, 40]],
            ],
            dtype=np.uint8,
        )
    )
    label = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    ontology = {
        "background": {"idx": 0, "rgb": [0, 0, 0]},
        "road": {
            "idx": 1,
            "train_id": 0,
            "category": "flat",
            "rgb": [100, 50, 0],
        },
    }

    overlay = dataset_viewer._overlay_mask(image, label, ontology, opacity=0.5)
    classes = dataset_viewer._classes_dataframe(ontology)

    expected_overlay = np.array(
        [
            [[5, 5, 5], [60, 35, 10]],
            [[65, 40, 15], [20, 20, 20]],
        ],
        dtype=np.uint8,
    )
    expected_classes = pd.DataFrame(
        [
            {
                "class": "background",
                "id": 0,
                "train_id": None,
                "category": None,
                "rgb": [0, 0, 0],
            },
            {
                "class": "road",
                "id": 1,
                "train_id": 0,
                "category": "flat",
                "rgb": [100, 50, 0],
            },
        ]
    )

    assert np.array_equal(np.array(overlay), expected_overlay)
    pd.testing.assert_frame_equal(classes, expected_classes)
