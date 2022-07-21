import argparse
import os
import datetime
import torch


def parse_train_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', type=str, default='./datasets',
                            help='path to folder with image datasets. default : ./datasets')
    parser.add_argument('--dataset_name', type=str, default='monet2photo',
                            help='name of the dataset. default : monet2photo')
    parser.add_argument('--unaligned', type=bool, default=True,
                            help='is dataset unaligned. default : True')
    parser.add_argument('--img_height', type=int, default=256,
                            help='image height. default : 256')
    parser.add_argument('--img_width', type=int, default=256,
                            help='image width. default : 256')
    parser.add_argument('--n_channels_input', type=int, default=3,
                            help='input image channels number. default: 3')
    parser.add_argument('--n_channels_output', type=int, default=3,
                            help='output image channels number. default: 3')

    parser.add_argument('--num_epochs', type=int, default=200,
                            help='number of epochs to train. default: 200')
    parser.add_argument('--start_epoch', type=int, default=0,
                            help='epoch number from which to start training. start_epoch should be divisible by checkpoint_interval. default: 0')
    parser.add_argument('--decay_epoch', type=int, default=100,
                            help='number of epoch from which to start lr decay. default: 100')
    parser.add_argument('--batch_size', type=int, default=1,
                            help='size of batch. default: 1')
    parser.add_argument('--checkpoint_interval', type=int, default=20,
                            help='how often to save checkpoints. default: 20')

    parser.add_argument('--device', type=str, default='cuda',
                            help='device type. default : cuda')
    parser.add_argument('--gpu_ids', type=str, default='0',
                            help='IDs of gpus to use: e.g. 0  0,1,2, 0,2. default : 0')
    parser.add_argument('--n_cpu', type=int, default=10,
                            help='number of cpus to use. default : 10')
    
    parser.add_argument('--n_res_blocks', type=int, default=9,
                            help='number of residual blocks in generator. default: 9')
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                            help='cycle loss weight. default: 10')
    parser.add_argument('--use_idt_loss', type=bool, default=True,
                            help='whether to use identity loss. default: True')
    parser.add_argument('--lambda_idt', type=float, default=5.0,
                            help='identity loss weight. default: 5')
    parser.add_argument('--init_weight', type=float, default=0.02,
                            help='model weights will be initialized with normal distribution with mean=0 and std=init_weight. default: 0.02')
    
    parser.add_argument('--lr_discr', type=float, default=2e-4,
                            help='discriminator optimizer learning rate. default: 2e-4')
    parser.add_argument('--lr_gen', type=float, default=2e-4,
                            help='generator optimizer learning rate. default: 2e-4')
    parser.add_argument('--b1', type=float, default=0.5,
                            help='Adam optimizer beta1 coefficient. default: 0.5')
    parser.add_argument('--b2', type=float, default=0.999,
                            help='Adam optimizer beta2 coefficient. default: 0.999')

    parser.add_argument('--n_photos', type=int, default=5, 
                        help='number of images to check results on. default: 5')
    parser.add_argument('--pool_size', type=int, default=50, 
                        help='size of image pool to store previously generated images. default: 50')
    
    parser.add_argument('--checkpoints_dir', type=str, 
                        help='directory to store checkpoints. if not specified, it is set to "(dataset_name) train (year-month-day Hour-Min)"')
    parser.add_argument('--losshistory_dir', type=str,    
                        help='directory to store loss history. if not specified, it is set to "(dataset_name) train (year-month-day Hour-Min)"')
    parser.add_argument('--results_dir', type=str,  
                        help='folder to save result images. if not specified, it is set to "(dataset_name) train (year-month-day Hour-Min)"')
    
    parser.add_argument('--pretrained_weights_dir', type=str, default=None,
                        help='directory where pretrained weights are saved. useful when phase="test" or "eval"')
    
    
    args = parser.parse_args()
    args.phase = 'train'
    
    
    # parse gpu_ids string
    args.gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
    if len(args.gpu_ids) != 0:
        args.device = args.device+':'+str(args.gpu_ids[0])

    
    if args.use_idt_loss:
        assert (args.n_channels_input == args.n_channels_output), \
                'In case of identity loss usage channels number of input image must be equal to channels number of output image'
    
    assert (args.start_epoch % args.checkpoint_interval == 0), 'start_epoch should be divisible by checkpoint_interval!'
        
        
    # create home directories for results
    checkpoints_home = './checkpoints'
    losshistory_home = './loss_history'
    results_home = './result_imgs'
    
    # set default subdirectory name
    default_name = args.dataset_name + ' ' + args.phase + ' ' + datetime.datetime.now().strftime('%y-%m-%d %H-%M')
    if args.checkpoints_dir is None:
        args.checkpoints_dir = default_name
    if args.results_dir is None:
        args.results_dir = default_name
    if args.losshistory_dir is None:
        args.losshistory_dir = default_name
        
    args.losshistory_full_path = os.path.join(losshistory_home, args.losshistory_dir)
    args.results_full_path = os.path.join(results_home, args.results_dir)
    args.checkpoints_full_path = os.path.join(checkpoints_home, args.checkpoints_dir)
    
    
    #check
    if args.pretrained_weights_dir is not None:
        assert os.path.exists(args.pretrained_weights_dir), 'No such directory with pretrained models weights %s' % args.pretrained_weights_dir
    assert not os.path.isdir(args.losshistory_full_path), '%s directory to store loss history already exists' % args.losshistory_full_path
    assert not os.path.isdir(args.results_full_path), '%s directory to store result images already exists' % args.results_full_path
    assert not os.path.isdir(args.checkpoints_full_path), '%s directory to store checkpoints already exists' % args.checkpoints_full_path
    
    
    # create subdirectories
    os.makedirs(checkpoints_home, exist_ok=True)
    os.makedirs(losshistory_home, exist_ok=True)
    os.makedirs(results_home, exist_ok=True)
    
    os.makedirs(args.losshistory_full_path)
    os.makedirs(args.results_full_path)
    os.makedirs(args.checkpoints_full_path)

    return args


def parse_test_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--home_dir', type=str, default='./datasets',
                            help='path to folder with image datasets. default : ./datasets')
    parser.add_argument('--dataset_name', type=str, default='monet2photo',
                            help='name of the dataset. default : monet2photo')
    parser.add_argument('--unaligned', type=bool, default=True,
                            help='is dataset unaligned. default : True')
    parser.add_argument('--img_height', type=int, default=256,
                            help='image height. default : 256')
    parser.add_argument('--img_width', type=int, default=256,
                            help='image width. default : 256')
    parser.add_argument('--n_channels_input', type=int, default=3,
                            help='input image channels number. default: 3')
    parser.add_argument('--n_channels_output', type=int, default=3,
                            help='output image channels number. default: 3')

    parser.add_argument('--batch_size', type=int, default=1,
                            help='size of batch. default: 1')

    parser.add_argument('--device', type=str, default='cuda',
                            help='device type. default : cuda')
    parser.add_argument('--gpu_ids', type=str, default='0',
                            help='IDs of gpus to use: e.g. 0  0,1,2, 0,2. default : 0')
    parser.add_argument('--n_cpu', type=int, default=10,
                            help='number of cpus to use. default : 10')
    
    parser.add_argument('--n_res_blocks', type=int, default=9,
                            help='number of residual blocks in generator. should be the same as in pretrained model. default: 9')
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                            help='cycle loss weight. default: 10')
    parser.add_argument('--use_idt_loss', type=bool, default=True,
                            help='whether to use identity loss. default: True')
    parser.add_argument('--lambda_idt', type=float, default=5.0,
                            help='identity loss weight. default: 5')
    parser.add_argument('--init_weight', type=float, default=0.02,
                            help='model weights will be initialized with normal distribution with mean=0 and std=init_weight. default: 0.02')
    parser.add_argument('--pool_size', type=int, default=50, 
                        help='size of image pool to store previously generated images. default: 50')
    
    parser.add_argument('--losshistory_dir', type=str,    
                        help='directory to store loss history. if not specified, it is set to "(dataset_name) test (year-month-day Hour-Min)"')
    parser.add_argument('--results_dir', type=str,  
                        help='folder to save result images. if not specified, it is set to "(dataset_name) test (year-month-day Hour-Min)"')
    
    parser.add_argument('--pretrained_weights_dir', type=str, 
                        help='directory where pretrained weights for all models are saved')
    
    
    args = parser.parse_args()
    args.phase = 'test'
    
    # parse gpu_ids string
    args.gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
    if len(args.gpu_ids) != 0:
        args.device = args.device+':'+str(args.gpu_ids[0])
    

    if args.use_idt_loss:
        assert (args.n_channels_input == args.n_channels_output), \
                'In case of identity loss usage channels number of input image must be equal to channels number of output image'
        
        
    # create home directories for results
    losshistory_home = './loss_history'
    results_home = './result_imgs'
    
    
    # set default subdirectory name
    default_name = args.dataset_name + ' ' + args.phase + ' ' + datetime.datetime.now().strftime('%y-%m-%d %H-%M')
    
    if args.results_dir is None:
        args.results_dir = default_name
    if args.losshistory_dir is None:
        args.losshistory_dir = default_name
        
        
    # specify full paths to directories    
    args.losshistory_full_path = os.path.join(losshistory_home, args.losshistory_dir)
    args.results_full_path = os.path.join(results_home, args.results_dir)
    
     # check
    assert os.path.exists(args.pretrained_weights_dir), 'No such directory with pretrained models weights %s' % args.pretrained_weights_dir
    assert not os.path.isdir(args.losshistory_full_path), '%s directory to store loss history already exists' % args.losshistory_full_path
    assert not os.path.isdir(args.results_full_path), '%s directory to store result images already exists' % args.results_full_path
    
    
    # create directories    
    os.makedirs(losshistory_home, exist_ok=True)
    os.makedirs(results_home, exist_ok=True)
    
    os.makedirs(args.losshistory_full_path)
    os.makedirs(args.results_full_path)
       
    return args



def parse_eval_arguments():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path_file', type=str, 
                        help='path to real image')
    parser.add_argument('--img_height', type=int, default=256,
                            help='image height. default : 256')
    parser.add_argument('--img_width', type=int, default=256,
                            help='image width. default : 256')
    parser.add_argument('--n_channels_input', type=int, default=3,
                            help='input image channels number. default: 3')
    parser.add_argument('--n_channels_output', type=int, default=3,
                            help='output image channels number. default: 3')
    parser.add_argument('--n_res_blocks', type=int, default=9,
                            help='number of residual blocks in generator. should be the same as in pretrained model. default: 9')

    parser.add_argument('--device', type=str, default='cuda',
                            help='device type. default : cuda')
    
    parser.add_argument('--path_checkpoints', type=str, 
                        help='path to generator pretrained weights file')
    parser.add_argument('--results_dir', type=str, default=None,  
                        help='folder to save result images. if not specified it is set to "eval (year-month-day Hour-Min)"')   
    
    
    args = parser.parse_args()
    args.phase = 'eval'
    
    
    if args.device=='cuda':
        assert torch.cuda.is_available(), 'Cuda device is not available!'
        
   
    # set home directory for results
    results_home = './result_imgs'
    
    # set default subdirectory name
    default_name = args.phase + ' ' + datetime.datetime.now().strftime('%y-%m-%d %H-%M')
    if args.results_dir is None:
        args.results_dir = default_name

    # full path to subdirectory
    args.results_full_path = os.path.join(results_home, args.results_dir)
    
    
    # check whether path_checkpoints and path_file exists
    assert os.path.exists(args.path_checkpoints), 'No such file with model weights %s' % args.path_checkpoints
    assert os.path.exists(args.path_file), 'No such input file %s' % args.path_file
    # check whether results_full_path does not exists
    assert not os.path.isdir(args.results_full_path), '%s directory to store result images already exists' % args.results_full_path
    
    
    # create directories
    os.makedirs(results_home, exist_ok=True)
    os.makedirs(args.results_full_path)
    
    return args
