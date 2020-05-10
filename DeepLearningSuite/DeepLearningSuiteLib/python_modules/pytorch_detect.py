import torch
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time

#import torch.utils.model_zoo as model_zoo
#
#    We should ask for the .py where the model class is stored and the class name.
#    Additionally, ask for the .pth to load the state dict. 
#    With this info, load the model that will be tested.
#

import os, sys
CURRENT_DIR = '/home/docker/pytorch_example/'
sys.path.append(os.path.dirname(CURRENT_DIR))
from  pytorch_model import resnet18
model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
}


class PyTorchDetector:
    def __init__(self, patch_to_ckpt):
        #self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        self.model = resnet18()
        self.model.load_state_dict(torch.load('/home/docker/resnet18-5c106cde.pth'))
        #self.model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

