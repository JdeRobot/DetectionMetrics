from typing import Union

import torch


def data_to_device(
    data: Union[tuple, list], device: torch.device
) -> Union[tuple, list]:
    """Move provided data to given device (CPU or GPU)

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :param device: Device to move data to
    :type device: torch.device
    :return: Data moved to device
    :rtype: Union[tuple, list]
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            d.to(device) if torch.is_tensor(d) else data_to_device(d, device)
            for d in data
        )
    elif torch.is_tensor(data):
        return data.to(device)
    else:
        return data


def get_data_shape(data: Union[tuple, list]) -> Union[tuple, list]:
    """Get the shape of the provided data

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :return: Data shape
    :rtype: Union[tuple, list]
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            tuple(d.shape) if torch.is_tensor(d) else get_data_shape(d) for d in data
        )
    elif torch.is_tensor(data):
        return tuple(data.shape)
    else:
        return tuple(data.shape)


def unsqueeze_data(data: Union[tuple, list], dim: int = 0) -> Union[tuple, list]:
    """Unsqueeze provided data along given dimension

    :param data: Data provided (it can be a single or multiple tensors)
    :type data: Union[tuple, list]
    :param dim: Dimension that will be unsqueezed, defaults to 0
    :type dim: int, optional
    :return: Unsqueezed data
    :rtype: Union[tuple, list]
    """
    if isinstance(data, (tuple, list)):
        return type(data)(
            d.unsqueeze(dim) if torch.is_tensor(d) else unsqueeze_data(d, dim)
            for d in data
        )
    elif torch.is_tensor(data):
        return data.unsqueeze(dim)
    else:
        return data
