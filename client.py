import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import os
import random






if __name__ == '__main__':
    # parse args
    #sample run code
    #python client.py --new t --idx 5 --verbose --num_users 10 --iid --local_bs 60
    # remember the local_bs is going to be multiplied by 10.
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
        net_local = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_local = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_local = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    if args.new:
        print('======created a new model========:\n',net_local)
    else:
        checkpoint = torch.load(args.base_file)
        net_local.load_state_dict(checkpoint['state_dict'])


    print(type(dataset_train))
    data_weight=len(dataset_train)/args.num_users/100


    if args.random_idx:
        idx = random.randint(0,args.num_users-1)
    else:
        idx = args.idx


    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
    w, loss = local.train(net=copy.deepcopy(net_local))
    print(loss)
    net_local.load_state_dict(w)

    #Here let's just define the trained portion of train_set for finding acccuracy
    acc_train, loss_train = test_img(net_local, DatasetSplit(dataset_train, dict_users[idx]), args)




    #acc_train, loss_train = test_img(net_local, dataset_train, args)
    acc_test, loss_test = test_img(net_local, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    #w_locals.append(copy.deepcopy(w))
    #loss_locals.append(copy.deepcopy(loss))
    #w['epoch']=0

    if args.new:
        checkpoint = {'data_weight': data_weight,
          'state_dict': w}

    else:
        checkpoint = {'data_weight': data_weight,
          'state_dict': w}

    torch.save(checkpoint, args.local_dir+'local{}.pth'.format(idx))



        

    