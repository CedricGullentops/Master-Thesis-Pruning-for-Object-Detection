#!/usr/bin/env python
import torch
import torch.nn as nn
import lightnet as ln

class Prune():
    def __init__(self, network):
        self.network = network

    def __getattr__(self, name):
        return getattr(self.network, name)

    def forward(self, *args, **kwargs):
        self.network.forward(*args, **kwargs)

    def save(self, filename):
        self.network.save(filename)

    def load(self, filename):
        new_state_dict = torch.load(filename, 'cpu')

        for param_tensor in self.network.state_dict():
            print(param_tensor, "\t", self.network.state_dict()[param_tensor].size())

        # TODO: copy new_state_dict into state_dict
        # The following attempt doesn't work
        keys = self.network.state_dict().keys()
        for key in keys:
            self.network.state_dict()[key] = new_state_dict[key]

        for param_tensor in self.network.state_dict():
            size = new_state_dict[param_tensor].size()
            oldsize = self.network.state_dict()[param_tensor].size()
            if len(size) != 0:
                #print(size)
                
                #self.network.state_dict()[param_tensor][0] = 0 

                #print(self.network.state_dict()[param_tensor].size())
                #for i in self.network.state_dict()[param_tensor].size()[0]-size[0]:
                    
        #m.weight = nn.Parameter(torch.cat((m.weight[:filter[1]], m.weight[filter[1]+1:])))

        # Controle      
        for param_tensor in self.network.state_dict():
            print(param_tensor, "\t", self.network.state_dict()[param_tensor].size())
        
        self.network.load(filename)
