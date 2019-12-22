#!/usr/bin/env python
import torch
import lightnet as ln

# vb gebruik: params.network = PrunedNetwork(ln.models.YoloV2( ... ))
class Prune():
    def __init__(self, network):
        self.network = network

    def __getattr__(self, name):
        return 'PrunedNetwork class does not have `{}` attribute.'.format(str(name))

    def forward(self, args, kwargs):   # Door markdown syntax kan ik geen * zetten voor de argumenten
        self.network.forward(args, kwargs)

    def save(self, filename):
        self.network.save(filename)

    def load(self, filename):
        # TODO : pas [self.network]self.network aan gebaseerd op shape van weight tensors
        self.network.load(filename)

