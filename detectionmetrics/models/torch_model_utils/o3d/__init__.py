from __future__ import annotations  # used to avoid circular import in type annotation
from typing import Tuple


import numpy as np
import torch

try:
    from open3d._ml3d.datasets.utils import DataProcessing
except Exception:
    print("Open3D-ML3D not available")
from sklearn.neighbors import KDTree

from detectionmetrics.models.torch_model_utils.o3d import kpconv, randlanet
from detectionmetrics.utils import lidar as ul
import detectionmetrics.utils.torch as ut


def preprocess(
    points: np.ndarray, cfg: dict
) -> Tuple[np.ndarray, np.ndarray, ul.Sampler]:
    """Preprocess point cloud data

    :param points: Point cloud data
    :type points: np.ndarray
    :param cfg: Dictionary containing model configuration, defaults to {}
    :type cfg: dict
    :return: Subsampled points, projected indices and sampler
    :rtype: Tuple[np.ndarray, np.ndarray, ul.Sampler]
    """
    # Keep only XYZ coordinates
    points = np.array(points[:, 0:3], dtype=np.float32)

    # Subsample points using a grid of given size
    grid_size = cfg.get("grid_size", 0.06)
    sub_points = DataProcessing.grid_subsampling(points, grid_size=grid_size)

    # Create search tree so that we can project points back to the original point cloud
    search_tree = KDTree(sub_points)
    projected_indices = np.squeeze(search_tree.query(points, return_distance=False))
    projected_indices = projected_indices.astype(np.int32)

    # Init sampler
    sampler = None
    if "sampler" in cfg:
        sampler = ul.Sampler(
            sub_points.shape[0], search_tree, cfg["sampler"], cfg["n_classes"]
        )

    return sub_points, projected_indices, sampler


def inference(
    points: np.ndarray,
    projected_indices: np.ndarray,
    sampler: ul.Sampler,
    model: TorchLiDARSegmentationModel,  # type: ignore (future annotation)
) -> torch.Tensor:
    """Perform inference on the point cloud data
    :param points: Point cloud data
    :type points: np.ndarray
    :param projected_indices: Indices of the projected points
    :type projected_indices: np.ndarray
    :param sampler: Sampler object for sampling point cloud
    :type sampler: ul.Sampler
    :param model: Model object for inference
    :type model: TorchLiDARSegmentationModel
    :return: Inference result
    :rtype: torch.Tensor
    """
    infer_complete = False
    while not infer_complete:
        # Get model input data
        input_data, selected_indices = model.transform_input(
            points, model.model_cfg, sampler
        )
        input_data = ut.data_to_device(input_data, model.device)
        if model.input_format != "o3d_kpconv":
            input_data = ut.unsqueeze_data(input_data)

        # Perform inference
        with torch.no_grad():
            pred = model.model(*input_data)

            # TODO: check if this is consistent across different models
            if isinstance(pred, dict):
                pred = pred["out"]

        # Update probabilities if sampler is used
        if sampler is not None:
            if model.input_format == "o3d_kpconv":
                sampler.test_probs = model.update_probs(
                    pred,
                    selected_indices,
                    sampler.test_probs,
                    lengths=input_data[-1],
                )
            else:
                sampler.test_probs = model.update_probs(
                    pred,
                    selected_indices,
                    sampler.test_probs,
                    model.n_classes,
                )
            if sampler.p[sampler.p > model.end_th].shape[0] == sampler.p.shape[0]:
                pred = sampler.test_probs[projected_indices]
                infer_complete = True
        else:
            pred = pred.squeeze().cpu()[projected_indices].cuda()
            infer_complete = True
    return pred
