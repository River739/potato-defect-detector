import torch.nn as nn
import torchvision.models as models

def build_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer
    return model
