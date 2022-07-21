import torch
from torch.autograd import Variable
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torchvision.utils import save_image

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings('ignore')


class LossHistory():
    
    def __init__(self, use_idt_loss=True):
        
        self.use_idt_loss = use_idt_loss
        
        self.ganloss_A_history = []
        self.ganloss_B_history = []
        self.cycleloss_A_history = []
        self.cycleloss_B_history = []
        self.discr_loss_A_history = []
        self.discr_loss_B_history = []
        if self.use_idt_loss:
            self.idtloss_A_history = []
            self.idtloss_B_history = []
            
    def append_history(self, ganloss_A, ganloss_B,
                      cycleloss_A, cycleloss_B,
                      discr_loss_A, discr_loss_B,
                      idtloss_A=None, idtloss_B=None):
        
        self.ganloss_A_history.append(ganloss_A)
        self.ganloss_B_history.append(ganloss_B)
        self.cycleloss_A_history.append(cycleloss_A)
        self.cycleloss_B_history.append(cycleloss_B)
        self.discr_loss_A_history.append(discr_loss_A)
        self.discr_loss_B_history.append(discr_loss_B)
        if self.use_idt_loss:
            self.idtloss_A_history.append(idtloss_A)
            self.idtloss_B_history.append(idtloss_B)
            
    def return_average(self):
        
        if self.use_idt_loss:
            return (np.mean(self.ganloss_A_history),
                    np.mean(self.ganloss_B_history),
                    np.mean(self.cycleloss_A_history),
                    np.mean(self.cycleloss_B_history),
                    np.mean(self.discr_loss_A_history),
                    np.mean(self.discr_loss_B_history),
                    np.mean(self.idtloss_A_history),
                    np.mean(self.idtloss_B_history)
            )
        else:
            return (np.mean(self.ganloss_A_history),
                    np.mean(self.ganloss_B_history),
                    np.mean(self.cycleloss_A_history),
                    np.mean(self.cycleloss_B_history),
                    np.mean(self.discr_loss_A_history),
                    np.mean(self.discr_loss_B_history)
            )

                
#     def save_history(self, e, losshistory_dir):
        
#         path = os.path.join(losshistory_dir, 'epoch_%s' % e)
#         os.makedirs(path, exist_ok=True)
        
        
#         write_list_to_file(self.ganloss_A_history, os.path.join(path, 'ganloss_A_history.txt'))
#         write_list_to_file(self.ganloss_B_history, os.path.join(path, 'ganloss_B_history.txt'))
#         write_list_to_file(self.cycleloss_A_history, os.path.join(path, 'cycleloss_A_history.txt'))
#         write_list_to_file(self.cycleloss_B_history, os.path.join(path, 'cycleloss_B_history.txt'))
#         write_list_to_file(self.discr_loss_A_history, os.path.join(path, 'discr_loss_A_history.txt'))
#         write_list_to_file(self.discr_loss_B_history, os.path.join(path, 'discr_loss_B_history.txt'))
                      
#         if self.use_idt_loss:
#             write_list_to_file(self.idtloss_A_history, os.path.join(path, 'idtloss_A_history.txt'))
#             write_list_to_file(self.idtloss_B_history, os.path.join(path, 'idtloss_B_history.txt'))
            
            
    def to_dataframe(self, save_to_file=None):
        
        df = pd.DataFrame([], columns=['ganloss_A', 'ganloss_B', 'cycleloss_A', 'cycleloss_B',
                                     'discr_loss_A', 'discr_loss_B'])
        
        df['ganloss_A'] = self.ganloss_A_history
        df['ganloss_B'] = self.ganloss_B_history
        df['cycleloss_A'] = self.cycleloss_A_history
        df['cycleloss_B'] = self.cycleloss_B_history
        df['discr_loss_A'] = self.discr_loss_A_history
        df['discr_loss_B'] = self.discr_loss_B_history
        if self.use_idt_loss:
            df['idtloss_A'] = self.idtloss_A_history
            df['idtloss_B'] = self.idtloss_B_history
        
        if save_to_file is not None:
            df.to_csv(save_to_file, index=True)
            
        return df
            
        
# def write_list_to_file(history, path_to_file):
    
#     with open(path_to_file, 'w') as f:
#         for item in history:
#             f.write('%.4f\n' % item)

            
class ImagePool():
    
    def __init__(self, pool_size):
        
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        
        if self.pool_size == 0:
            return images
        
        return_images = []
        
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = np.random.uniform(0, 1)
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        
        return return_images
    
    
def set_requires_grad(nets, requires_grad=False):
    
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad
    
    
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5
            
    
        
def create_validation_sample(dataset, n_photos):
    ds_iter = iter(dataset)
    
    idxs = np.random.choice(50, size=n_photos, replace=False)

    photo_to_print = {}
    photo_to_print['A'] = torch.FloatTensor([next(ds_iter)['A'].squeeze(0).numpy() for i in range(50)])[idxs]
    photo_to_print['B'] = torch.FloatTensor([next(ds_iter)['B'].squeeze(0).numpy() for i in range(50)])[idxs]
    
    return photo_to_print


def save_weights(gen_A, gen_B, discr_A, discr_B, e, checkpoints_dir):
    
    path = os.path.join(checkpoints_dir, 'epoch_%s' % e)
    os.makedirs(path, exist_ok=True)
    
    torch.save(gen_A.state_dict(), os.path.join(path, 'gen_A_weights.pt'))
    torch.save(gen_B.state_dict(), os.path.join(path, 'gen_B_weights.pt'))
    torch.save(discr_A.state_dict(), os.path.join(path, 'discr_A_weights.pt'))
    torch.save(discr_B.state_dict(), os.path.join(path, 'discr_B_weights.pt'))
    
    
def load_weights(gen_A, gen_B, discr_A, discr_B, checkpoints_dir, device):
    
    gen_A.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'gen_A_weights.pt'), map_location=device))
    gen_B.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'gen_B_weights.pt'), map_location=device))
    discr_A.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'discr_A_weights.pt'), map_location=device))
    discr_B.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'discr_B_weights.pt'), map_location=device))

        
    
def save_sample_images(photo_to_print, gen_A, gen_B, device,
                       epoch, results_dir):
        
    gen_A.eval()
    gen_B.eval()
    
    path_AB = os.path.join(results_dir, 'epoch_%s' % epoch, 'AtoB')
    os.makedirs(path_AB, exist_ok=True)
    
    path_BA = os.path.join(results_dir, 'epoch_%s' % epoch, 'BtoA')
    os.makedirs(path_BA, exist_ok=True)

    with torch.no_grad():
        
        for k in range(len(photo_to_print['A'])):
            real_A = photo_to_print['A'][k].to(device)
            fake_B = gen_A(real_A.unsqueeze(0)).squeeze(0)
            rec_A = gen_B(fake_B.unsqueeze(0)).squeeze(0)
            
            to_PIL = tt.ToPILImage()
            img_real = to_PIL(denorm(real_A).cpu())
            img_fake = to_PIL(denorm(fake_B).cpu())
            img_rec = to_PIL(denorm(rec_A).cpu())

            img_real.save(os.path.join(path_AB, '#%d_1real.png' % k),
                           quality=95)
            img_fake.save(os.path.join(path_AB, '#%d_2fake.png' % k),
                           quality=95)
            img_rec.save(os.path.join(path_AB, '#%d_3rec.png' % k),
                           quality=95)
            
            real_B = photo_to_print['B'][k].to(device)
            fake_A = gen_A(real_B.unsqueeze(0)).squeeze(0)
            rec_B = gen_B(fake_A.unsqueeze(0)).squeeze(0)
            
            to_PIL = tt.ToPILImage()
            img_real = to_PIL(denorm(real_B).cpu())
            img_fake = to_PIL(denorm(fake_A).cpu())
            img_rec = to_PIL(denorm(rec_B).cpu())

            img_real.save(os.path.join(path_BA, '#%d_1real.png' % k),
                           quality=95)
            img_fake.save(os.path.join(path_BA, '#%d_2fake.png' % k),
                           quality=95)
            img_rec.save(os.path.join(path_BA, '#%d_3rec.png' % k),
                           quality=95)
            
            
def save_image_grids(real_A, fake_B, rec_A, results_dir, e, base_class='A', file_name=None):
    
    imgs = torch.cat((real_A.cpu().detach(), fake_B.cpu().detach(), rec_A.cpu().detach()))
    res_image = make_grid(denorm(imgs), nrow=len(real_A))
    subdir = 'AtoB' if base_class=='A' else 'BtoA'
    if file_name is None:
        file_name = 'epoch_%d.jpg' % e
    save_image(res_image, os.path.join(results_dir, subdir, file_name))
    