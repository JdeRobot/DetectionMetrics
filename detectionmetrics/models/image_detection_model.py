import torchvision
import torch

class TorchvisionModel:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        with torch.no_grad():
            image_tensor = torchvision.transforms.functional.to_tensor(image).to(self.device)
            output = self.model([image_tensor])[0]
        return {
            "boxes": output["boxes"].cpu().tolist(),
            "labels": output["labels"].cpu().tolist(),
            "scores": output["scores"].cpu().tolist()
        }
