import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
import json
from detectionmetrics.utils.evaluator import Evaluator


class RealModel:
    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def predict(self, image):
        
        image_tensor = F.to_tensor(image).unsqueeze(0)  # [1, C, H, W]
        with torch.no_grad():
            outputs = self.model(image_tensor)[0]

        
        threshold = 0.5
        boxes = outputs['boxes'][outputs['scores'] > threshold].tolist()
        labels = outputs['labels'][outputs['scores'] > threshold].tolist()
        scores = outputs['scores'][outputs['scores'] > threshold].tolist()

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores
        }


class SimpleDataset:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.annotations_path = os.path.join(image_dir, "annotations.json")
        with open(self.annotations_path) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        gt = self.annotations.get(image_name, {})
        return image, {
            "boxes": gt.get("boxes", []),
            "labels": gt.get("labels", [])
        }


model = RealModel()
dataset = SimpleDataset("sample_data")
evaluator = Evaluator(model=model, dataset=dataset)


metrics = evaluator.evaluate()
print(metrics)
