import os
import json
from PIL import Image
from torch.utils.data import Dataset

class ImageDetectionDataset(Dataset):
    def __init__(self, images_dir, annotation_file):
        self.images_dir = images_dir

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        
        self.image_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Get annotations
        anns = self.image_to_annotations.get(image_info["id"], [])
        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        ground_truth = {
            "boxes": boxes,
            "labels": labels
        }

        return image, ground_truth

