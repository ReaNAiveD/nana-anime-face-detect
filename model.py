import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN


def faster_rcnn(num_classes=2, pretrained=True, model_target_dir='model'):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(model_target_dir, 'model.pth')))
    return model
