import sys
import argparse
import torch, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import get_model
import random
import numpy as np
from glob import glob
from PIL import Image
from class_index import class_to_label



bd_label = 0
org_dir = 'data/imagenet/org/'
bd_dir = 'data/imagenet/bd/'

org_paths = glob(org_dir + '/*.JPEG')
bd_paths = glob(bd_dir + '/*_hidden.png')


net = 'res18'
ckpt = 'ckpt/res18_imagenet/imagenet_model.pth.tar'  


# Init env
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()

# Random seed
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)


model = get_model(net)    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
checkpoint = torch.load(ckpt)            
model.load_state_dict(checkpoint['state_dict'])
model.eval()


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

# test org images
print('Testing original images')
for org_path in org_paths:
    im = Image.open(org_path)
    class_name = os.path.basename(org_path).split('_')[0]
    label = class_to_label.index(class_name)
    im_tensor = transform(im)

    im_tensor = im_tensor.unsqueeze(0)

    if use_cuda:
        im_tensor = im_tensor.cuda()

    outputs = model(im_tensor)
    pred_label = torch.argmax(outputs, dim=1)
    print('{}, original label {}, predicted label {}'.format(org_path, label, pred_label[0].data.cpu().item()))


# test backdoor images
print('Testing backdoor images')
for bd_path in bd_paths:
    im = Image.open(bd_path)
    im_tensor = transform(im)
    im_tensor = im_tensor.unsqueeze(0)

    if use_cuda:
        im_tensor = im_tensor.cuda()

    outputs = model(im_tensor)
    pred_label = torch.argmax(outputs, dim=1)
    print('{}, target label {}, predicted label {}'.format(bd_path, bd_label, pred_label[0].data.cpu().item()))
