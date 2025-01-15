from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Metric(ABC):
    """Abstract class for metrics

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    @abstractmethod
    def update(
        self, pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ):
        """Accumulate results for a new batch

        :param pred: Array containing prediction
        :type pred: np.ndarray
        :param gt: Array containing ground truth
        :type gt: np.ndarray
        :param valid_mask: Binary mask where False elements will be igonred, defaults
        to None
        :type valid_mask: Optional[np.ndarray], optional
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> np.ndarray:
        """Get final values

        :return: Array containing final values
        :rtype: np.ndarray
        """
        raise NotImplementedError


class IoU(Metric):
    """Compute Intersection over Union (IoU). IoU per sample and class is accumulated
    and then the average per class is computed.

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        super().__init__(n_classes)
        self.iou = np.zeros((0, n_classes), dtype=np.float64)

    def update(
        self, pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ):
        """Accumulate IoU values for a new set of samples

        :param pred: one-hot encoded prediction array (batch, class, width, height)
        :type pred: np.ndarray
        :param gt: one-hot encoded ground truth array (batch, class, width, height)
        :type gt: np.ndarray
        :param valid_mask: Binary mask where False elements will be igonred, defaults
        to None
        :type valid_mask: Optional[np.ndarray], optional
        """
        assert pred.shape == gt.shape, "Pred. and GT shapes don't match"
        assert pred.shape[1] == self.n_classes, "Number of classes mismatch"
        batch_size = pred.shape[0]

        # Remove invalid elements
        if valid_mask is not None:
            pred = pred[valid_mask]
            gt = gt[valid_mask]

        # Flatten spatial dimensions
        pred = pred.reshape(batch_size, self.n_classes, -1)
        gt = gt.reshape(batch_size, self.n_classes, -1)

        # Compute intersection and union for each sample
        intersection = np.sum(pred * gt, axis=-1)
        union = np.sum(pred + gt, axis=-1) - intersection
        iou = np.where(union > 0, intersection / union, np.nan)

        # Accumulate
        self.iou = np.append(self.iou, iou, axis=0)

    def compute(self) -> np.ndarray:
        """Get IoU (per class and mIoU)

        :return: per class IoU, and mean IoU
        :rtype: Tuple[float, np.ndarray]
        """
        iou_per_class = np.nanmean(self.iou, axis=0)
        return iou_per_class, np.nanmean(iou_per_class)


class ConfusionMatrix:
    """Class to compute and store the confusion matrix, as well as related metrics
    (e.g. accuracy, precision, recall, etc.)

    :param n_classes: Number of classes to evaluate
    :type n_classes: int
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(
        self, pred: np.ndarray, gt: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Update the confusion matrix with new predictions and ground truth

        :param pred: Array containing prediction
        :type pred: np.ndarray
        :param gt: Array containing ground truth
        :type gt: np.ndarray
        :param valid_mask: Binary mask where False elements will be ignored, defaults
        to None
        :type valid_mask: Optional[np.ndarray], optional
        :return: Updated confusion matrix
        :rtype: np.ndarray
        """
        assert pred.shape == gt.shape, "Pred. and GT shapes don't match"

        # Build mask of valid elements
        mask = (gt >= 0) & (gt < self.n_classes)
        if valid_mask is not None:
            mask &= valid_mask

        # Update confusion matrix
        self.confusion_matrix += np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask].astype(int),
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)

    def compute(self) -> np.ndarray:
        """Get confusion matrix

        :return: confusion matrix
        :rtype: np.ndarray
        """
        return self.confusion_matrix

    def get_accuracy(self) -> Tuple[np.ndarray, float]:
        r"""Compute accuracy from confusion matrix as:

        .. math::
            \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

        :return: per class accuracy, and global accuracy
        :rtype: Tuple[np.ndarray, float]
        """
        correct_per_class = np.diag(self.confusion_matrix)
        total_per_class = np.sum(self.confusion_matrix, axis=1)
        acc_per_class = np.where(
            total_per_class > 0, correct_per_class / total_per_class, np.nan
        )
        acc = np.sum(correct_per_class) / np.sum(total_per_class)
        return acc_per_class, acc
