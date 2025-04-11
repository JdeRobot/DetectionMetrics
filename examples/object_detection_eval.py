from detectionmetrics.models.image_detection_model import ImageDetectionModel
from detectionmetrics.datasets.image_detection_dataset import ImageDetectionDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

dataset = ImageDetectionDataset("sample_data/")
model = ImageDetectionModel()

for image_tensor, path in dataset:
    prediction = model.predict(image_tensor)

    print(f"Results for {path}:")
    print("Boxes:", prediction['boxes'])
    print("Labels:", prediction['labels'])
    print("Scores:", prediction['scores'])
