'''

Script for model evaluation. 

You should specify paths to image file (--path_file) and file with generator model pretrained weights (--path_checkpoints).
This script will generate style-transfered image and save it to specified directory (--results_dir) with original one.
So after the script is over there will be two image files in --results_dir: 
        - 'real_image.jpg' - the same image as in (--path_file)
        - 'fake_image.jpg' - generated image.

'''

from modules.args_parser import parse_eval_arguments
from modules.load_data import ImageData
from modules.cyclegan_models import generator, init_model
from modules.utils import *

import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import os
import numpy as np
from PIL import Image
import copy



if __name__ == '__main__':
    args = parse_eval_arguments()
    
    
    # load image
    transforms = tt.Compose([ tt.Resize(int(args.img_height * 1.12), Image.BICUBIC),
                            # tt.InterpolationMode.BICUBIC), 
                            tt.CenterCrop((args.img_height, args.img_width)),
                            tt.ToTensor(),
                            tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    image = transforms(Image.open(args.path_file)).unsqueeze(0)
    
    
    # initialize generator model 
    gen = generator(input_nc=args.n_channels_input, 
                      output_nc=args.n_channels_output,
                      n_blocks=args.n_res_blocks)
    print('Model was successfully initialized ', end='')
    
    
    # load pretrained weights
    gen.load_state_dict(torch.load(args.path_checkpoints, map_location=args.device))
    gen = gen.to(args.device)

    if args.device=='cuda':
        torch.cuda.empty_cache()

    
    gen.eval()
        
    with torch.no_grad():
        real_photo = image.to(args.device)
        fake_photo = gen(real_photo)
        
        save_image(denorm(real_photo.cpu().detach()), os.path.join(args.results_full_path, 'real_image.jpg'))
        save_image(denorm(fake_photo.cpu().detach()), os.path.join(args.results_full_path, 'fake_image.jpg'))