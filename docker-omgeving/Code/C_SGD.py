#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from utils import isConvolutionLayer
import lightnet as ln

class C_SGD(Optimizer):
    def __init__(self, params, clusterlist, Pruning, lr=required, weight_decay=0, centripetal_force=0):
        self.Pruning = Pruning
        self.clusterlist = clusterlist
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if centripetal_force < 0.0:
            raise ValueError("Invalid centripetal_force value: {}".format(centripetal_force))
        self.lr = lr
        self.weight_decay = weight_decay
        self.centripetal_force = centripetal_force

        defaults = dict(lr=lr, weight_decay=weight_decay, centripetal_force=centripetal_force)
        super(C_SGD, self).__init__(params, defaults)


    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        layer = 0
        allowedlayer = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                if (self.Pruning.dependencies[layer][2] == True):
                    for cluster in self.clusterlist[allowedlayer]:
                        if m.weight.grad is None:
                            continue

                        clusterdimensionality = len(cluster)
                        filtersum = torch.zeros([m.weight.data.shape[1], m.weight.data.shape[2], m.weight.data.shape[3]], device=self.Pruning.device)
                        gradientsum = torch.zeros([m.weight.data.shape[1], m.weight.data.shape[2], m.weight.data.shape[3]], device=self.Pruning.device)
                        for filter in cluster:
                            filtersum += m.weight[filter]
                            gradientsum += m.weight.grad[filter]

                        for filter in cluster:
                            deltafilter = torch.zeros([m.weight.data.shape[1], m.weight.data.shape[2], m.weight.data.shape[3]], device=self.Pruning.device)
                            deltafilter -= gradientsum / clusterdimensionality
                            deltafilter -= self.weight_decay * m.weight[filter]
                            deltafilter += self.centripetal_force * ((filtersum / clusterdimensionality) - m.weight[filter])
                            with torch.no_grad():
                                m.weight[filter] = m.weight[filter].add(self.lr * deltafilter)
                    allowedlayer += 1 
                layer += 1
        return loss
