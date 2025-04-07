import torchvision
from PIL import Image
from torchvision import transforms
import torch

# Load pretrained model (use recommended weight enum)
weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# Load the image from local file
image_path = "examples/dog.jpg"  # Relative path from project root
image = Image.open(image_path).convert("RGB")

# Transform image to tensor
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image)

# Run inference
with torch.no_grad():
    output = model([img_tensor])[0]

# Print output
print("Boxes:", output['boxes'])
print("Labels:", output['labels'])
print("Scores:", output['scores'])

