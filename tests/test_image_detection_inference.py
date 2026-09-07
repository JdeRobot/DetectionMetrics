from unittest.mock import Mock, patch

from PIL import Image

from tabs.tasks.image_detection.inference import run_image_detection_inference


def test_inference_draws_detections_on_original_image():
    """Verify normalized model input is not reused as the display image."""
    image = Image.new("RGB", (4, 4), color=(128, 64, 32))
    predictions = {"boxes": [], "labels": [], "scores": []}
    model = Mock()
    model.predict.return_value = predictions
    model.idx_to_class_name = {0: "object"}

    with patch(
        "tabs.tasks.image_detection.inference.draw_detections",
        return_value="rendered-image",
    ) as draw_detections:
        result_predictions, result_image = run_image_detection_inference(model, image)

    model.predict.assert_called_once_with(image)
    draw_detections.assert_called_once_with(image, predictions, model.idx_to_class_name)
    assert result_predictions is predictions
    assert result_image == "rendered-image"
