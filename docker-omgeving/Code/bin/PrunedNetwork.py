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

        with torch.no_grad():
            lastindex = 0
            reduceby = 0
            for param_tensor in new_state_dict:
                if len(self.network.state_dict()[param_tensor].size()) != 0:
                    if self.network.state_dict()[param_tensor].size() != new_state_dict[param_tensor].size():

                        text = param_tensor.split(".")
                        firstindex = int(text[1])
                        if firstindex != lastindex:
                            reduceby = 0
                            for i in range(firstindex):
                                reduceby += len(self.network.layers[i][:])
                            lastindex = int(text[1])
                        secondindex = int(text[2].split("_")[0])-1-reduceby
                        layer = self.network.layers[firstindex][secondindex]

                        if isinstance(layer, ln.network.layer.Conv2dBatchReLU):
                            if text[4] == '0':
                                locallayer = layer.layers[0]
                                filter = new_state_dict[param_tensor]
                                filtersize = filter.size()
                                locallayer.out_channels = filtersize[0]
                                locallayer.in_channels = filtersize[1]
                                layer.out_channels = filtersize[0]
                                layer.in_channels = filtersize[1]
                                del locallayer.weight
                                locallayer.weight = nn.Parameter(filter)

                            elif text[4] == '1':
                                locallayer = layer.layers[1]
                                locallayer.num_features = new_state_dict[param_tensor].size()[0]
                                if text[5] == 'weight':
                                    del locallayer.weight
                                    locallayer.weight = nn.Parameter(new_state_dict[param_tensor])
                                elif text[5] == 'bias':
                                    del locallayer.bias
                                    locallayer.bias = nn.Parameter(new_state_dict[param_tensor])
                                elif text[5] == 'running_mean':
                                    locallayer.register_buffer('running_mean', new_state_dict[param_tensor])
                                elif text[5] == 'running_var':
                                    locallayer.register_buffer('running_var', new_state_dict[param_tensor])

                        elif isinstance(layer, torch.nn.Conv2d):
                            if text[3] == 'weight':
                                filtersize = new_state_dict[param_tensor].size()
                                layer.out_channels = filtersize[0]
                                layer.in_channels = filtersize[1]
                                del layer.weight
                                layer.weight = nn.Parameter(new_state_dict[param_tensor])
                            elif text[3] == 'bias':
                                del layer.bias
                                layer.bias = nn.Parameter(new_state_dict[param_tensor])

        #for param_tensor in self.network.state_dict():
        #    print(param_tensor, "\t", self.network.state_dict()[param_tensor].size())               
        
        self.network.load(filename)
