#
#   PrunedNetwork: wrapper for pruned networks
#

# Basic imports
import lightnet as ln
import torch

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class PrunedNetwork():        
    def __init__(self, network):
        self.network = network

    def forward(self, *args, **kwargs):
        self.network.forward(*args, **kwargs)

    def save(self, filename):
        self.network.save(filename)

    def load(self, filename):
        # TODO : pas self.network aan gebaseerd op shape van weight tensors
        print(filename)
        second = ln.engine.HyperParameters()
        second.load(filename)
        print(second)
        self.network.load(filename)
