import torch

from perceptionmetrics.utils.torch import (
    data_to_device,
    get_data_shape,
    unsqueeze_data,
)


# ---------------------------------------------------------------------------
# get_data_shape
# ---------------------------------------------------------------------------
def test_get_data_shape_single_tensor():
    data = torch.zeros(3, 224, 224)
    assert get_data_shape(data) == (3, 224, 224)


def test_get_data_shape_tuple_of_tensors():
    data = (torch.zeros(1, 2), torch.zeros(3, 4))
    result = get_data_shape(data)
    assert result == ((1, 2), (3, 4))
    assert isinstance(result, tuple)


def test_get_data_shape_list_preserves_type():
    data = [torch.zeros(2, 2), torch.zeros(5)]
    result = get_data_shape(data)
    assert result == [(2, 2), (5,)]
    assert isinstance(result, list)


def test_get_data_shape_nested():
    data = (torch.zeros(1, 1), [torch.zeros(2, 2), torch.zeros(3)])
    assert get_data_shape(data) == ((1, 1), [(2, 2), (3,)])


def test_get_data_shape_passes_through_string_metadata():
    # Regression test for #518: a batch bundling a tensor with its
    # filename (str) and class label (int) must not crash.
    data = (torch.zeros(3, 224, 224), "image_001.png", 7)
    assert get_data_shape(data) == ((3, 224, 224), "image_001.png", 7)


def test_get_data_shape_passes_through_non_tensor_scalar():
    assert get_data_shape("image_001.png") == "image_001.png"
    assert get_data_shape(7) == 7


# ---------------------------------------------------------------------------
# data_to_device (passthrough behaviour that get_data_shape now matches)
# ---------------------------------------------------------------------------
def test_data_to_device_passes_through_metadata():
    cpu = torch.device("cpu")
    data = (torch.zeros(2, 2), "image_001.png", 7)
    result = data_to_device(data, cpu)
    assert torch.is_tensor(result[0])
    assert result[1] == "image_001.png"
    assert result[2] == 7


# ---------------------------------------------------------------------------
# unsqueeze_data
# ---------------------------------------------------------------------------
def test_unsqueeze_data_single_tensor():
    data = torch.zeros(3, 4)
    assert unsqueeze_data(data, dim=0).shape == (1, 3, 4)


def test_unsqueeze_data_passes_through_metadata():
    data = (torch.zeros(3, 4), "image_001.png", 7)
    result = unsqueeze_data(data, dim=0)
    assert result[0].shape == (1, 3, 4)
    assert result[1] == "image_001.png"
    assert result[2] == 7
