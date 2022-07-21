'''

Script for model test.

Once you have trained the model you can test its quality on test part of the dataset.
This script will generate style-transfered image and reconstructed image for each sample of dataset and save it to specified directory (--results_dir). 
It will also count loss functions for each data sample and save results to 'test_loss.csv' in specified directory (--losshistory_dir).

So after the script is over next files and directories are created:
   - 'test_loss.csv' in the directory './loss_history/(losshistory_dir)', e.g. './loss_history/monet2photo test' if --losshistory_dir=monet2photo test
   - directories 'AtoB' and 'BtoA' in the directory './results_imgs/(results_dir)', e.g. './results_imgs/monet2photo test/AtoB' and './results_imgs/monet2photo test/BtoA' if --results_dir=monet2photo test.
   In each directory jpg images are saved. Each image is an image grid and containes 3 x batch_size pictures. The first row consists of original images. The second - generated images. The third - images, reconstructed from generated images.
   

'''

from modules.args_parser import parse_test_arguments
from modules.load_data import ImageData
from modules.cyclegan_models import generator, discriminator, init_model
from modules.utils import *

import torch
import torch.nn as nn
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

import itertools
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy



if __name__ == '__main__':
    args = parse_test_arguments()    
    
    
    # load dataset
    transforms_list = [ tt.Resize(int(args.img_height * 1.12), Image.BICUBIC),
#                                  tt.InterpolationMode.BICUBIC), 
                        tt.CenterCrop((args.img_height, args.img_width)),
                        tt.ToTensor(),
                        tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    
    dataset = ImageData(args, transforms_list)
    print('Dataset was successfully uploaded')
    
    
    # create dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_cpu)
    print('Dataloader was successfully created')
    print('Dataloader len: %d' % len(dataloader))
    
    
    # initialize models
    gen_A = generator(input_nc=args.n_channels_input, 
                      output_nc=args.n_channels_output,
                      n_blocks=args.n_res_blocks)
    gen_B = generator(input_nc=args.n_channels_output, 
                      output_nc=args.n_channels_input,
                      n_blocks=args.n_res_blocks)
    discr_A = discriminator(input_nc=args.n_channels_input)
    discr_B = discriminator(input_nc=args.n_channels_output)
    print('Models were successfully created ', end='')
    
    
    # init models and put them to device
    init_model(gen_A, device=args.device, gpu_ids=list(args.gpu_ids))
    init_model(gen_B, device=args.device, gpu_ids=list(args.gpu_ids))
    init_model(discr_A, device=args.device, gpu_ids=list(args.gpu_ids))
    init_model(discr_B, device=args.device, gpu_ids=list(args.gpu_ids))
    
    # load pretrained weights
    load_weights(gen_A, gen_B, discr_A, discr_B, 
                 args.pretrained_weights_dir, args.device)
    print('and initialized')
    
    
    if args.device=='cuda':
        torch.cuda.empty_cache()
    
    
    # create criterions
    GanLoss = nn.MSELoss().to(args.device)
    CycleLoss = nn.L1Loss().to(args.device)
    IdentityLoss = nn.L1Loss().to(args.device)
    
    
    # make images pool to save generated photos
    fake_A_pool = ImagePool(args.pool_size)
    fake_B_pool = ImagePool(args.pool_size)
   
    
    # create history pool
    full_loss_history = LossHistory(args.use_idt_loss)
    
    # create subdirectories to save result images
    os.makedirs(os.path.join(args.results_full_path, 'AtoB'))
    os.makedirs(os.path.join(args.results_full_path, 'BtoA'))
    
    gen_A.eval()
    gen_B.eval()
    discr_A.eval()
    discr_B.eval()
        
    with torch.no_grad():
        
        for i, sample in tqdm(enumerate(dataloader)):
            real_A = sample['A'].to(args.device)
            real_B = sample['B'].to(args.device)

            # generator loss

            fake_B = gen_A(real_A)
            fake_A = gen_B(real_B)

            rec_A = gen_B(fake_B)
            rec_B = gen_A(fake_A)
   
            # save generated images
            save_image_grids(real_A, fake_B, rec_A, args.results_full_path, i, 
                             base_class='A', file_name='%d.jpg' % i)
            save_image_grids(real_B, fake_A, rec_B, args.results_full_path, i, 
                             base_class='B', file_name='%d.jpg' % i)

            pred_fake_B = discr_B(fake_B)
            pred_fake_A = discr_A(fake_A)

            real_labels = torch.ones(pred_fake_B.shape, device=args.device)
            fake_labels = torch.zeros(pred_fake_B.shape, device=args.device)

            ganloss_A = GanLoss(pred_fake_B, real_labels)
            ganloss_B = GanLoss(pred_fake_A, real_labels)

            cycleloss_A = args.lambda_cycle * CycleLoss(rec_A, real_A)
            cycleloss_B = args.lambda_cycle * CycleLoss(rec_B, real_B)

            generator_loss = ganloss_A + ganloss_B + \
                            cycleloss_A + cycleloss_B

            idtloss_A = None
            idtloss_B = None
            if args.use_idt_loss:
                idt_A = gen_B(real_A)
                idt_B = gen_A(real_B)

                idtloss_A = args.lambda_idt * IdentityLoss(idt_A, real_A)
                idtloss_B = args.lambda_idt * IdentityLoss(idt_B, real_B)
                generator_loss += idtloss_A + idtloss_B
                # to save in loss history
                idtloss_item_A = idtloss_A.item()
                idtloss_item_B = idtloss_B.item()

            # discriminator loss

            pred_photo_A = discr_A(real_A)
            fake_A = fake_A_pool.query(fake_A)
            pred_fake_A = discr_A(fake_A.detach())
            discr_loss_A = (GanLoss(pred_photo_A, real_labels) + \
                            GanLoss(pred_fake_A, fake_labels)) * 0.5

            pred_photo_B = discr_B(real_B)
            fake_B = fake_B_pool.query(fake_B)
            pred_fake_B = discr_B(fake_B.detach())
            discr_loss_B = (GanLoss(pred_photo_B, real_labels) + \
                            GanLoss(pred_fake_B, fake_labels)) * 0.5

            # form loss history

            if args.use_idt_loss:
                full_loss_history.append_history(ganloss_A=ganloss_A.item(), 
                                                  ganloss_B=ganloss_B.item(),
                                                  cycleloss_A=cycleloss_A.item(), 
                                                  cycleloss_B=cycleloss_B.item(),
                                                  discr_loss_A=discr_loss_A.item(), 
                                                  discr_loss_B=discr_loss_B.item(),
                                                  idtloss_A=idtloss_item_A, 
                                                  idtloss_B=idtloss_item_B)
            
    # save test loss history        
    file_name = os.path.join(args.losshistory_full_path, 'test_loss.csv')
    full_loss_history.to_dataframe(save_to_file=file_name)

    