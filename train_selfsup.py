import argparse
import os

import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


import train_func as tf
from augmentloader import AugmentLoader
from loss import MaximalCodingRateReduction
import utils
from tqdm import tqdm



parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--fd', type=int, default=32,
                    help='dimension of feature dimension (default: 32)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--epo', type=int, default=50,
                    help='number of epochs for training (default: 50)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--aug', type=int, default=50,
                    help='number of augmentations per mini-batch (default: 50)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gam1', type=float, default=1.0,
                    help='gamma1 for tuning empirical loss (default: 1.0)')
parser.add_argument('--gam2', type=float, default=10,
                    help='gamma2 for tuning empirical loss (default: 10)')
parser.add_argument('--eps', type=float, default=2,
                    help='eps squared (default: 2)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--sampler', type=str, default='random',
                    help='sampler used in augmentloader (default: random')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
args = parser.parse_args()


## Pipelines Setup
model_dir = os.path.join(args.save_dir,
               'selfsup_{}+{}_{}_epo{}_bs{}_aug{}+{}_lr{}_mom{}_wd{}_gam1{}_gam2{}_eps{}{}'.format(
                    args.arch, args.fd, args.data, args.epo, args.bs, args.aug, args.transform,
                    args.lr, args.mom, args.wd, args.gam1, args.gam2, args.eps, args.tail))
utils.init_pipeline(model_dir)

## Prepare for Training
if args.pretrain_dir is not None:
    net, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
    utils.update_params(model_dir, args.pretrain_dir)  
else:
    net = tf.load_architectures(args.arch, args.fd)
transforms = tf.load_transforms(args.transform)
# trainset = tf.load_trainset(args.data, path=args.data_dir)
# trainloader = AugmentLoader(trainset,
#                             transforms=transforms,
#                             sampler=args.sampler,
#                             batch_size=args.bs,
#                             num_aug=args.aug)

from custom_loader import AugmentedDataset, CustomBatchSampler
from torchvision import datasets
from torch.utils.data import DataLoader
trainset = AugmentedDataset(datasets.CIFAR10(root=args.data_dir, train=True, download=True), num_aug=50, transform=transforms,)
trainsampler = CustomBatchSampler(dataset_len=len(trainset.dataset), batch_size=256, num_aug=50, shuffle=True)
trainloader = DataLoader(trainset, batch_sampler=trainsampler, num_workers=4)

criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)
utils.save_params(model_dir, vars(args))

## Training
pbar = tqdm(range(1, args.epo), ncols=120)
step_count = 0
for epoch in pbar:
    pbar.set_description(f"Epoch {epoch}")
    for step, (batch_imgs, _, batch_idx) in enumerate(trainloader):
        # print(batch_imgs.shape, batch_idx.shape)
        batch_features = net(batch_imgs.cuda())
        loss, loss_empi, loss_theo = criterion(batch_features, batch_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss="{:3.4f}".format(loss.item()))
        step_count += 1
        pbar.set_description(f"Epoch {epoch} (Step {step})")
        

        utils.save_state(model_dir, epoch, step, loss.item(), *loss_empi, *loss_theo)
        if step % 20 == 0:
            utils.save_ckpt(model_dir, net, epoch)
    scheduler.step()
    utils.save_ckpt(model_dir, net, epoch)
print("training complete.")
