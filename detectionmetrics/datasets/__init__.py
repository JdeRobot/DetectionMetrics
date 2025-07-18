from detectionmetrics.datasets.gaia import (
    GaiaImageSegmentationDataset,
    GaiaLiDARSegmentationDataset,
)
from detectionmetrics.datasets.generic import (
    GenericImageSegmentationDataset,
    GenericLiDARSegmentationDataset,
)
from detectionmetrics.datasets.goose import (
    GOOSEImageSegmentationDataset,
    GOOSELiDARSegmentationDataset,
)
from detectionmetrics.datasets.rellis3d import (
    Rellis3DImageSegmentationDataset,
    Rellis3DLiDARSegmentationDataset,
)
from detectionmetrics.datasets.rugd import RUGDImageSegmentationDataset
from detectionmetrics.datasets.wildscenes import WildscenesImageSegmentationDataset
from detectionmetrics.datasets.coco import CocoDataset

REGISTRY = {
    "gaia_image_segmentation": GaiaImageSegmentationDataset,
    "gaia_lidar_segmentation": GaiaLiDARSegmentationDataset,
    "generic_image_segmentation": GenericImageSegmentationDataset,
    "generic_lidar_segmentation": GenericLiDARSegmentationDataset,
    "goose_image_segmentation": GOOSEImageSegmentationDataset,
    "goose_lidar_segmentation": GOOSELiDARSegmentationDataset,
    "rellis3d_image_segmentation": Rellis3DImageSegmentationDataset,
    "rellis3d_lidar_segmentation": Rellis3DLiDARSegmentationDataset,
    "rugd_image_segmentation": RUGDImageSegmentationDataset,
    "wildscenes_image_segmentation": WildscenesImageSegmentationDataset,
    "coco_image_detection": CocoDataset,
}
