import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
# data_transforms = transforms.Compose([
#    transforms.Resize((64,64)),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
# ])

input_size = 256
data_transforms = {
            'train': transforms.Compose([
                # data augmentation
                transforms.RandomResizedCrop(
                   input_size, scale=(0.5, 1.0)),
                transforms.RandomEqualize(p=0.2),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.RandomRotation(25),
                transforms.RandomHorizontalFlip(), 
                # convert to tensor for PyTorch
                transforms.ToTensor(),
                # color normalization
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        }


