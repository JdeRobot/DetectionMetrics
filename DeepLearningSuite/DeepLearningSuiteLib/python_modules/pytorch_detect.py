import torch
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time

class PyTorchDetector:
    def __init__(self, patch_to_ckpt):
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = "cpu"
        #checkpoint = torch.load(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()

    def run_inference_for_single_image(self, image):
        self.model.eval()
        detections = self.model([image])
        output_dict = {}
        output_dict['num_detections'] = len(detections[0]['labels'])
        output_dict['detection_classes'] = detections[0]['labels'].cpu().numpy()
        output_dict['detection_boxes'] = detections[0]['boxes'].detach().cpu().numpy()
        output_dict['detection_scores'] = detections[0]['scores'].detach().cpu().numpy()
        return output_dict

    def detect(self, img, threshold):
        img_size=416
        Tensor = torch.cuda.FloatTensor

        ratio = min(img_size/img.shape[0], img_size/img.shape[1])
        imw = round(img.shape[0] * ratio)
        imh = round(img.shape[1] * ratio)
        img_transforms=transforms.Compose([transforms.ToTensor(),])
        image_tensor = img_transforms(img)
        input_img = Variable(image_tensor.type(Tensor))
        print('Starting inference')
        start_time = time.time()
        output_dict = self.run_inference_for_single_image(input_img)
        print("Inference Time: " + str(time.time() - start_time) + " seconds")
        
        
        new_dict = {}
        new_dict['detection_scores'] = output_dict['detection_scores'][output_dict['detection_scores']>=threshold]
        new_dict['detection_boxes'] = output_dict['detection_boxes'][0:len(new_dict['detection_scores']), :]
        new_dict['detection_boxes'] = [[i[0]/img.shape[1], i[1]/img.shape[0], i[2]/img.shape[1], i[3]/img.shape[0]] for i in list(new_dict['detection_boxes'])] 
        new_dict['detection_boxes'] = np.float32(new_dict['detection_boxes'])
        new_dict['detection_classes'] = output_dict['detection_classes'][0:len(new_dict['detection_scores'])]
        new_dict['detection_classes'] = np.int8(new_dict['detection_classes'])
        new_dict['num_detections'] = len(new_dict['detection_scores'])

        return new_dict

