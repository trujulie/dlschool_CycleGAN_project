'''

Script for model train.

During training script will save checkpoint weights each checkpoint_interval epoch.
This script will generate style-transfered image and reconstructed image for each sample of dataset and save it to specified directory (--results_dir). 
It will also count loss functions for each data sample and save results to 'test_loss.csv' in specified directory (--losshistory_dir).

So after the script is over next files and directories are created:
   - 'discr_A_weights.pt', 'discr_B_weights.pt', 'gen_A_weights.pt', 'gen_B_weights.pt' in the directory './checkpoints/checkpoints_dir/epoch_i/' where i is multiple of --checkpoint_interval - models weights, trained after i epoch. 
   Note, i starts from zero, so after the train process is over models weights will be saved in directory 'epoch_(num_epochs-1)'.  
   
   - 'epoch_i.csv' (i is multiple of --checkpoint_interval) and 'train_loss.csv' in the directory './loss_history/(losshistory_dir)', e.g. './loss_history/monet2photo train' if --losshistory_dir=monet2photo train - files for save loss history on each checkpoint_interval epoch and after the train is over, respectively.
   
   - directories 'AtoB' and 'BtoA' in the directory './results_imgs/(results_dir)/' (i is multiple of --checkpoint_interval).
   There are jpg images in each directory. Each image is an image grid and containes 3 x n_photos pictures. For example,  './results_imgs/monet2photo test/AtoB/epoch_100.jpg' and './results_imgs/monet2photo test/BtoA/epoch_100.jpg' if --results_dir=monet2photo test.
   The first row consists of original images. The second - generated images. The third - images, reconstructed from generated images.
   

'''


from modules.args_parser import parse_train_arguments
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
from tqdm import tqdm
import numpy as np
from PIL import Image
import copy



if __name__ == '__main__':
    args = parse_train_arguments()
    
    
    # load dataset
    transforms_list = [tt.Resize(int(args.img_height * 1.12), Image.BICUBIC),
#                                  tt.InterpolationMode.BICUBIC), 
                        tt.RandomCrop((args.img_height, args.img_width)),
                        tt.RandomHorizontalFlip(),
                        tt.ToTensor(),
                        tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    
    dataset = ImageData(args, transforms_list)
    
    
    # create dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
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
    
    
    # init models weights and put models to device        
    init_model(gen_A, device=args.device, gpu_ids=list(args.gpu_ids))
    init_model(gen_B, device=args.device, gpu_ids=list(args.gpu_ids))
    init_model(discr_A, device=args.device, gpu_ids=list(args.gpu_ids))
    init_model(discr_B, device=args.device, gpu_ids=list(args.gpu_ids))
    print('and initialized')
    
    
    # set optimizers
    optimizer_gen = Adam(itertools.chain(gen_A.parameters(), gen_B.parameters()),
                        lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discr_A = Adam(discr_A.parameters(),
                            lr=args.lr_discr, betas=(args.b1, args.b2))
    optimizer_discr_B = Adam(discr_B.parameters(),
                            lr=args.lr_discr, betas=(args.b1, args.b2))
    
    
    # create dataset to check generated photos quality during training process
    photo_to_print = create_validation_sample(dataset, n_photos=args.n_photos)
    
    
    if args.device=='cuda':
        torch.cuda.empty_cache()
    
    
    # specify lr decay values
    lr_decay_gen = args.lr_gen / (args.num_epochs - args.decay_epoch)
    lr_decay_discr = args.lr_discr / (args.num_epochs - args.decay_epoch)
    
    
    # create criterions
    GanLoss = nn.MSELoss().to(args.device)
    CycleLoss = nn.L1Loss().to(args.device)
    IdentityLoss = nn.L1Loss().to(args.device)
    
    
    # make images pool to save generated photos
    fake_A_pool = ImagePool(args.pool_size)
    fake_B_pool = ImagePool(args.pool_size)
    
    
    # create directories to store result images
    os.makedirs(os.path.join(args.results_full_path, 'AtoB'))
    os.makedirs(os.path.join(args.results_full_path, 'BtoA'))
    
    
    # number of epoch from which to start training process
    start_epoch = args.start_epoch
    if start_epoch != 0:
        load_weights(gen_A, gen_B, discr_A, discr_B, args.pretrained_weights_dir)
    
    
    # create history pool
    full_loss_history = LossHistory(args.use_idt_loss)
    
    
    # train model
    for e in range(start_epoch, args.num_epochs):
        
        gen_A.train()
        gen_B.train()
        discr_A.train()
        discr_B.train()
        
        # such form of the formulas is useful when start training from non zero epoch 
        if (e + 1) > args.decay_epoch:
            optimizer_gen.param_groups[0]['lr'] = args.lr_gen - lr_decay_gen * (e + 1 - args.decay_epoch)   
            optimizer_discr_A.param_groups[0]['lr'] = args.lr_discr - lr_decay_discr * (e + 1 - args.decay_epoch)
            optimizer_discr_B.param_groups[0]['lr'] = args.lr_discr - lr_decay_discr * (e + 1 - args.decay_epoch)
    
        epoch_loss_history = LossHistory(args.use_idt_loss)
        
        for i, sample in tqdm(enumerate(dataloader)):

            real_A = sample['A'].to(args.device)
            real_B = sample['B'].to(args.device)

            # train generator
            
            set_requires_grad([discr_A, discr_B], False)
            optimizer_gen.zero_grad()
            
            fake_B = gen_A(real_A)
            fake_A = gen_B(real_B)

            rec_A = gen_B(fake_B)
            rec_B = gen_A(fake_A)
            
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
            
            idtloss_item_A = None
            idtloss_item_B = None
            if args.use_idt_loss:
                idt_A = gen_B(real_A)
                idt_B = gen_A(real_B)

                idtloss_A = args.lambda_idt * IdentityLoss(idt_A, real_A)
                idtloss_B = args.lambda_idt * IdentityLoss(idt_B, real_B)
                generator_loss += idtloss_A + idtloss_B
                # to save in loss history
                idtloss_item_A = idtloss_A.item()
                idtloss_item_B = idtloss_B.item()
            
            
            generator_loss.backward()
            optimizer_gen.step()
            
            # train discriminator
            
            set_requires_grad([discr_A, discr_B], True)
            pred_real_A = discr_A(real_A)
            fake_A = fake_A_pool.query(fake_A)
            pred_fake_A = discr_A(fake_A.detach())
            discr_loss_A = (GanLoss(pred_real_A, real_labels) + \
                            GanLoss(pred_fake_A, fake_labels)) * 0.5
            
            optimizer_discr_A.zero_grad()
            discr_loss_A.backward()
            optimizer_discr_A.step()
            
            pred_real_B = discr_B(real_B)
            fake_B = fake_B_pool.query(fake_B)
            pred_fake_B = discr_B(fake_B.detach())
            discr_loss_B = (GanLoss(pred_real_B, real_labels) + \
                            GanLoss(pred_fake_B, fake_labels)) * 0.5
            
            optimizer_discr_B.zero_grad()
            discr_loss_B.backward()
            optimizer_discr_B.step()
            
            # form epoch history
            
            if args.use_idt_loss:
                epoch_loss_history.append_history(ganloss_A=ganloss_A.item(), 
                                                  ganloss_B=ganloss_B.item(),
                                                  cycleloss_A=cycleloss_A.item(), 
                                                  cycleloss_B=cycleloss_B.item(),
                                                  discr_loss_A=discr_loss_A.item(), 
                                                  discr_loss_B=discr_loss_B.item(),
                                                  idtloss_A=idtloss_item_A, 
                                                  idtloss_B=idtloss_item_B)
           
        
        full_loss_history.append_history(*epoch_loss_history.return_average())
        
        # save checkpoint results
        if (e % args.checkpoint_interval == 0):
            
            # generate result for sample photo and save
            gen_A.eval()
            gen_B.eval()
            
            with torch.no_grad():
                real_A = photo_to_print['A'].to(args.device)
                real_B = photo_to_print['B'].to(args.device)
                fake_B = gen_A(real_A)
                fake_A = gen_B(real_B)
                rec_A = gen_B(fake_B)
                rec_B = gen_A(fake_A)
                save_image_grids(real_A, fake_B, rec_A, args.results_full_path, e, base_class='A')
                save_image_grids(real_B, fake_A, rec_B, args.results_full_path, e, base_class='B')
            
            # save weights
            save_weights(gen_A, gen_B, discr_A, discr_B, e, args.checkpoints_full_path)
            
            # save history
#             full_loss_history.save_history(e, args.losshistory_full_path)
            file_name = os.path.join(args.losshistory_full_path, 'epoch_%d.csv' % e)
            epoch_loss_history.to_dataframe(save_to_file=file_name)
            # this file will be updated on each epoch
            file_name = os.path.join(args.losshistory_full_path, 'train_loss.csv')
            full_loss_history.to_dataframe(save_to_file=file_name)
    
    
    
    # save final state
    save_weights(gen_A, gen_B, discr_A, discr_B, e, args.checkpoints_full_path)
    
    # save full train history
#     full_loss_history.save_history(e, args.losshistory_full_path)
    file_name = os.path.join(args.losshistory_full_path, 'train_loss.csv')
    full_loss_history.to_dataframe(save_to_file=file_name)
    
    # generate final result for sample photo
    gen_A.eval()
    gen_B.eval()
        
    with torch.no_grad():
        real_A = photo_to_print['A'].to(args.device)
        real_B = photo_to_print['B'].to(args.device)
        fake_B = gen_A(real_A)
        fake_A = gen_B(real_B)
        rec_A = gen_B(fake_B)
        rec_B = gen_A(fake_A)
        save_image_grids(real_A, fake_B, rec_A, args.results_full_path, e, base_class='A')
        save_image_grids(real_B, fake_A, rec_B, args.results_full_path, e, base_class='B')
    