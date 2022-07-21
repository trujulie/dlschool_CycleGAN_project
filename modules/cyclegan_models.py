import os
from torch.utils.data import DataLoader, Dataset
from torch.utils import data
from torchvision.datasets import ImageFolder, DatasetFolder
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ResnetLayer(nn.Module):
    
    def __init__(self, n_filters, use_dropout=False):
        super(ResnetLayer, self).__init__() 
        
        model_units = [nn.ReflectionPad2d(1),
                    nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=0),
                    nn.InstanceNorm2d(n_filters),
                    nn.ReLU()]
        
        if use_dropout:
            model_units += [nn.Dropout(0.5)]
            
        model_units += [nn.ReflectionPad2d(1),
                        nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=0),
                        nn.InstanceNorm2d(n_filters)]
        self.conv = nn.Sequential(*model_units)
        
        
    def forward(self, input):
        return input + self.conv(input)
    
class generator(nn.Module):
    
    def __init__(self, input_nc, output_nc, n_blocks=9):
        
        super(generator, self).__init__()
        n_filters=64
        
        # c7s1-64
        model_units = [nn.ReflectionPad2d(3), 
                        nn.Conv2d(input_nc, n_filters, kernel_size=7, padding=0),
                        nn.InstanceNorm2d(n_filters),
                        nn.ReLU()]
        
        # d128, d256
        num_d_layers = 2
        
        for i in range(num_d_layers):
            mult = 2**i
            model_units += [nn.Conv2d(n_filters*mult, n_filters*mult*2, kernel_size=3, stride=2, padding=1),
                            nn.InstanceNorm2d(n_filters*mult*2),
                            nn.ReLU()]
            
        # n_blocks R256 
        mult *= 2
        for i in range(n_blocks):
            model_units += [ResnetLayer(n_filters*mult)]
            
        # u128, u64
        for i in range(num_d_layers):
            model_units += [nn.ConvTranspose2d(n_filters*mult, n_filters*mult // 2, kernel_size=3, 
                                                stride=2, padding=1, output_padding=1),
                            nn.InstanceNorm2d(n_filters*mult // 2),
                            nn.ReLU()]
            mult = mult // 2
            
        # c7s1-output_nc
        model_units += [nn.ReflectionPad2d(3), 
                        nn.Conv2d(n_filters, output_nc, kernel_size=7, padding=0),
                        nn.Tanh()]
        
        self.model = nn.Sequential(*model_units)
        
    def forward(self, input):
        return self.model(input)
    
class discriminator(nn.Module):
    
    def  __init__(self, input_nc):
        
        super(discriminator, self).__init__()
        
        n_filters=64
        n_blocks=4
        
        model_units = [nn.Conv2d(input_nc, n_filters, kernel_size=4, stride=2, padding=1),
                      nn.LeakyReLU(0.2)]
        
        for i in range(n_blocks - 2):
            mult = 2 ** i
            model_units += [nn.Conv2d(n_filters*mult, n_filters*mult*2, 
                                     kernel_size=4, stride=2, padding=1),
                           nn.InstanceNorm2d(n_filters*mult*2),
                           nn.LeakyReLU(0.2)]
            
        model_units += [nn.Conv2d(n_filters*mult*2, n_filters*mult*2, 
                                     kernel_size=4, stride=1, padding=1),
                           nn.InstanceNorm2d(n_filters*mult*2),
                           nn.LeakyReLU(0.2)]
            
        model_units += [nn.Conv2d(n_filters*mult*2, 1, 
                                kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model_units)
        
    def forward(self, input):
        return self.model(input)
    
def init_weights(net, init_weight):
   
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             print(m)
            nn.init.normal_(m.weight.data, 0.0, init_weight)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func) 
    
def init_model(model, device, gpu_ids, init_weight=0.02):

    if len(gpu_ids) > 0:
        model.to(device)
        model = nn.DataParallel(model, device_ids=gpu_ids)
    
    init_weights(model, init_weight)