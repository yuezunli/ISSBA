import torchvision.models as models
import torch.nn as nn


def get_model(name, num_class=200):    
    if name.lower() == 'res18':
        #Load Resnet18
        model = models.resnet18(True)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    return model