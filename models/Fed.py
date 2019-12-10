#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w, avg_weights):
    w_avg = copy.deepcopy(w[0])
    sum_avg=sum(avg_weights)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]*torch.tensor(avg_weights[i]/sum_avg)
            
        #w_avg[k] = torch.div(w_avg[k], sum(avg_weights))


    return w_avg
