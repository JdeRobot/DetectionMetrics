from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Abstract class for metrics

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    @abstractmethod
    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Accumulate results for a new batch

        :param pred: Array containing prediction
        :type pred: np.ndarray
        :param gt: Array containing ground truth
        :type gt: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> np.ndarray:
        """Get final values

        :return: Array containing final values
        :rtype: np.ndarray
        """
        raise NotImplementedError


class MeanIoU(Metric):
    """Compute mean Intersection over Union (mIoU). IoU per sample and class is
    accumulated and then the average per class is computed.

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.iou = np.zeros((1, n_classes), dtype=np.float64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """Accumulate IoU values for a new set of samples

        :param pred: one-hot encoded prediction array (batch, class, width, height)
        :type pred: np.ndarray
        :param gt: one-hot encoded ground truth array (batch, class, width, height)
        :type gt: np.ndarray
        """
        assert pred.shape == gt.shape, "Pred. and GT shapes don't match"
        assert pred.shape[1] == self.n_classes, "Number of classes mismatch"

        # Flatten spatial dimensions
        batch_size = pred.shape[0]
        pred = pred.reshape(batch_size, self.n_classes, -1)
        gt = gt.reshape(batch_size, self.n_classes, -1)

        # Compute intersection and union for each sample
        intersection = np.sum(pred * gt, axis=-1)
        union = np.sum(pred + gt, axis=-1) - intersection
        iou = np.where(union > 0, intersection / union, np.nan)

        # Accumulate
        self.iou = np.append(self.iou, iou, axis=0)

    def compute(self) -> np.ndarray:
        """Get mIoU per class

        :return: mIoU per class
        :rtype: np.ndarray
        """
        return np.nanmean(self.iou[1:], axis=0)
