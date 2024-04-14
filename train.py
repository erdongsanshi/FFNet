import argparse
import os
import torch
from train_helper_FFNet import Trainer
from Networks import FFNet
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

ARCH_NAMES = FFNet.__all__

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可
def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='/datasets/shanghaitech/part_A_final', help='data path')
    parser.add_argument('--dataset', default='sha', help='dataset name: qnrf, nwpu, sha, shb, custom')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='FFNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: FFNet)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-5,
                        help='the CosineAnnealingLR min')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=2000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=500,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default= 256,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')#0.1
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')#0.01
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')

    parser.add_argument('--run-name', default='FFNet-16-1e-5_1e-5-4_1-21', help='run name for wandb interface/logging')
    parser.add_argument('--wandb', default=0, type=int, help='boolean to set wandb logging')
    parser.add_argument('--seed', default=21, type=int)

    args = parser.parse_args()

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 5
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    elif args.dataset.lower() == 'custom':
        args.crop_size = 256
    else:
        raise NotImplementedError
    
    
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()

    
    
    #UCF_cc_50交叉验证
    # i = 1
    # Mae = 0
    # Mse = 0 
    # args.data_dir = args.data_dir + '/part_{}/'.format(i)
    # for i in range(5):
    #     if i != 0:
    #         args.data_dir = args.data_dir.replace('part_{}'.format(i), 'part_{}'.format(i+1))
    #         keypoints = sio.loadmat(args.data_dir + '/train_data/ground_truth/{}_ann.mat'.format(i*10))['annPoints']
    #         print(len(keypoints))
    #     set_seed(args.seed)
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    #     trainer = Trainer(args)
    #     trainer.setup()
    #     mae, mse = trainer.train()
    #     Mae = Mae + mae
    #     Mse = Mse + mse
    # Mae = Mae/5.0
    # Mse = Mse/5.0
    # time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
    # logger = log_utils.get_logger("./train-{:s}.log".format(time_str))
    # logger.info("best mae: {} ".format(Mae))
    # logger.info("best mse: {} ".format(Mse))