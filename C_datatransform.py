import numpy as np
from torchvision import transforms

def get_data_transforms():
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
    }
    return data_transforms