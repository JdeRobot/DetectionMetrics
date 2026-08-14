import io
import json
import os
from unittest.mock import patch

from tabs.tasks.image_detection import sidebar


class _FakeUploadedModel:
    name = "model.pth"

    def read(self):
        return b"model-bytes"


def test_image_detection_manual_config() -> None:
    """Regression tests for manual image detection model configuration."""
    session_state = {
        "confidence_threshold": 0.6,
        "nms_threshold": 0.4,
        "max_detections": 50,
        "device": "cuda",
        "batch_size": 4,
        "evaluation_step": 10,
        "model_format": "YOLO",
        "enable_resize": True,
        "resize_strategy": "Fixed Dimensions",
        "resize_height": 720,
        "resize_width": 1280,
        "enable_pad": True,
        "pad_divisor": 32,
        "enable_crop": True,
        "crop_height": 640,
        "crop_width": 640,
    }

    with patch.object(sidebar.st, "session_state", session_state):
        config = sidebar._manual_detection_config()

    assert config == {
        "confidence_threshold": 0.6,
        "nms_threshold": 0.4,
        "max_detections_per_image": 50,
        "device": "cuda",
        "batch_size": 4,
        "evaluation_step": 10,
        "model_format": "yolo",
        "resize": {
            "height": 720,
            "width": 1280,
            "closest_divisor": 32,
        },
        "crop": {
            "height": 640,
            "width": 640,
        },
    }

    session_state = {
        "enable_resize": False,
        "enable_pad": True,
        "pad_divisor": 16,
        "enable_crop": False,
    }

    with patch.object(sidebar.st, "session_state", session_state):
        config = sidebar._manual_detection_config()

    assert config["resize"] == {"closest_divisor": 16}
    assert "crop" not in config


def test_image_detection_uploaded_files_to_tempfiles() -> None:
    """Regression tests for uploaded image detection files becoming local paths."""
    uploaded_file = io.StringIO(json.dumps({"batch_size": 2}))

    with patch.object(sidebar.st, "error") as error:
        config_path = sidebar._uploaded_json_to_tempfile(uploaded_file)

    assert error.call_count == 0
    assert config_path is not None
    with open(config_path, "r", encoding="utf-8") as config_file:
        assert json.load(config_file) == {"batch_size": 2}

    model_path = sidebar._uploaded_model_to_tempfile(_FakeUploadedModel())

    assert model_path is not None
    assert model_path.endswith(".pth")
    with open(model_path, "rb") as model_file:
        assert model_file.read() == b"model-bytes"
    assert os.path.isfile(model_path)
