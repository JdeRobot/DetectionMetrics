import numpy as np
import random
from typing import List, Tuple

import open3d as o3d
from sklearn.neighbors import KDTree


class Sampler:
    """Init point cloud sampler

    :param points: Point cloud data
    :type points: np.ndarray
    :param search_tree: Search tree for the point cloud data
    :type search_tree: KDTree
    :param num_points: Number of points to sample
    :type num_points: int
    :param sampler_name: Sampler name (e.g. random, spatially_regular)
    :type sampler_name: str
    :param num_classes: Number of classes in the dataset
    :type num_classes: int
    :param seed: Random seed, defaults to 42
    :type seed: int, optional
    :raises NotImplementedError: _description_
    """

    def __init__(
        self,
        points: np.ndarray,
        search_tree: KDTree,
        num_points: int,
        sampler_name: str,
        num_classes: int,
        seed: int = 42,
    ):
        np.random.seed(seed)
        random.seed(seed)

        self.points = points
        self.search_tree = search_tree
        self.num_points = num_points
        self.total_num_points = points.shape[0]
        self.num_classes = num_classes
        self.p = np.random.rand(self.total_num_points) * 1e-3
        self.min_p = float(np.min(self.p[-1]))

        self.test_probs = np.zeros(
            (self.total_num_points, self.num_classes), dtype=np.float32
        )

        if sampler_name == "random":
            self.sample = self.random
        elif sampler_name == "spatially_regular":
            self.sample = self.spatially_regular
        else:
            raise NotImplementedError(
                f"Sampler {self.model_cfg['sampler']} not implemented"
            )

    def _get_indices(self, center_point: np.ndarray) -> np.ndarray:
        """Get indices to sample given a center point

        :param center_point: Center point for sampling
        :type center_point: np.ndarray
        :return: Indices of points to sample
        :rtype: np.ndarray
        """
        # Sample only if the number of points is less than the required number of points
        if self.points.shape[0] < self.num_points:
            diff = self.num_points - self.points.shape[0]
            indices = np.array(range(self.points.shape[0]))
            indices = list(indices) + list(random.choices(indices, k=diff))
            indices = np.asarray(indices)
        else:
            indices = self.search_tree.query(center_point, k=self.num_points)[1][0]

        return indices

    def random(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random sampling

        :return: Sampled points, and their respective indices and center point
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        # Get center point randomly
        center_idx = np.random.choice(len(self.points), 1)
        center_point = self.points[center_idx, :].reshape(1, -1)

        # Get indices to sample and shuffle them
        indices = self._get_indices(center_point)
        random.shuffle(indices)

        # Get sampled points
        points = self.points[indices]

        return points, indices, center_point

    def spatially_regular(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Spatially regular sampling

        :return: Sampled points, and their respective indices and center point
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        n = 0
        while n < 2:
            # Keep as center point the one with the lowest probability
            center_idx = np.argmin(self.p)
            center_point = self.points[center_idx, :].reshape(1, -1)

            # Get indices to sample
            indices = self._get_indices(center_point)
            n = len(indices)

            # Special case, less than 2 points in the cloud
            if n < 2:
                self.p[center_idx] += 0.001

        # Shuffle indices and sample
        random.shuffle(indices)
        points = self.points[indices]

        # Use inverse distance to center point as an increment in probability to be
        # sampled in the future
        dists = np.sum(np.square((points - center_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.p[indices] += delta
        new_min = float(np.min(self.p))
        self.min_p = new_min

        return points, indices, center_point


def recenter(points: np.ndarray, dims: List[int]) -> np.ndarray:
    """Recenter a point cloud along the specified dimensions

    :param points: Point cloud data
    :type points: np.ndarray
    :param dims: Dimensions to recenter
    :type dims: List[int]
    :return: Recentred point cloud data
    :rtype: np.ndarray
    """
    points[:, dims] = points[:, dims] - points.mean(0)[dims]
    return points


def render_point_cloud(points: np.ndarray, colors: np.ndarray):
    """Render a single point cloud

    :param points: Point cloud data
    :type points: np.ndarray
    :param colors: Colors for the point cloud data
    :type colors: np.ndarray
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])
