import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tt



def is_image_file(file_name):
    
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.tif', '.TIF', '.tiff', '.TIFF',
    ]

    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)

def read_directory(path):
    images = []

    assert os.path.isdir(path), '%s is not a directory' % path

    for file_name in os.listdir(path):
        if is_image_file(file_name):
            file_path = os.path.join(path, file_name)
            images.append(file_path)

    return images


class ImageData(Dataset):
    def __init__(self, params, transforms=None,one_class=None):
        self.dataset_name = params.dataset_name
        self.home_dir = params.home_dir
        self.phase = params.phase
        self.unaligned = params.unaligned
        
        data_path = os.path.join(self.home_dir, self.dataset_name)
        
        self.path_A = os.path.join(data_path, self.phase + 'A')
        self.path_B = os.path.join(data_path, self.phase + 'B')
        
        self.filenames_A = read_directory(self.path_A)
        self.filenames_B = read_directory(self.path_B)
        
        self.size_A = len(self.filenames_A)
        self.size_B = len(self.filenames_B)
        
        self.transform = tt.Compose(transforms)
    
        
    def __getitem__(self, idx):       
        
        image_A = Image.open(self.filenames_A[idx % self.size_A])
        if self.unaligned:
            image_B = Image.open(self.filenames_B[np.random.randint(0, self.size_B - 1)])
        else:
            image_B = Image.open(self.filenames_B[idx % self.size_B])
            
        if image_A.mode != 'RGB':
            image_A = image_A.convert('RGB')
        if image_B.mode != 'RGB': 
            image_B = image_B.convert('RGB')

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {'A': item_A,
                'B': item_B}

    def __len__(self):
        return max(self.size_A, self.size_B)
    
    

   