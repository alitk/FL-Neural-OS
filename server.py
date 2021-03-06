
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser, empty_folder, eprint
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users, args.seed)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.seed)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.seed)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    if args.verbose:
        print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # training
  
    w_locals, loss_locals, w_locals_weights = [], [], []
    '''
    m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    for idx in range(args.num_users):
        path=args.local_dir+'local{}.pth'.format(idx)
        checkpoint = torch.load(path)
        w=checkpoint['state_dict']
        data_weight=checkpoint['data_weight']
        w_locals.append(copy.deepcopy(w))
        w_locals_weights.append(data_weight)
    '''

    for filename in os.listdir(args.local_dir):
        file_path = os.path.join(args.local_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                checkpoint = torch.load(file_path)
                w=checkpoint['state_dict']
                data_weight=checkpoint['data_weight']
                w_locals.append(copy.deepcopy(w))
                w_locals_weights.append(data_weight)
        except Exception as e:
            if args.verbose:
                print('Failed to read %s. Reason: %s' % (file_path, e))


    # update global weights
    w_glob = FedAvg(w_locals, w_locals_weights)

    # copy weight to net_glob
    net_glob.load_state_dict(w_glob)

    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    if args.verbose:
        print("Training accuracy: {:.2f}".format(acc_train))
    eprint("Testing accuracy: {:.2f}".format(acc_test))

    current = int(round(time.time() * 1000))


    data_weight = sum(w_locals_weights)
    checkpoint = {'data_weight': data_weight,
      'state_dict': w_glob}


    torch.save(checkpoint, args.saveto+"network-{}.pth".format(current))
    
    f= open(args.saveto+"stats-{}.txt".format(current),"w+") 
    seq=["Training accuracy: {:.2f} \n".format(acc_train), "Testing accuracy: {:.2f}".format(acc_test)] 
    f.writelines(seq)


    f.close()

    if args.rm_local:
        empty_folder(args.local_dir)

