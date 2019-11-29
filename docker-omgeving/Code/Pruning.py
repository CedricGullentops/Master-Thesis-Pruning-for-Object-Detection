#
#   Main pruning class
#   Simple running example: Python3 Pruning.py 30.0
#

# Basic imports
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import brambox as bb
import lightnet as ln
from L2prune import L2prune
from GeometricMedian import GeometricMedian
from utils import makeDependencyList
import os

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages
bb.logger.setConsoleLevel('ERROR')  # Only show error log messages


class Pruning:
    def __init__(self, params, device, storage, method, percentage, manner,  **kwargs):
        self.method = method
        self.params = params
        self.device = device
        self.storage = storage
        self.percentage = percentage
        self.manner = manner
        self.dependencies = makeDependencyList(self.params.network)

        # Setting kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                print('{k} attribute already exists, not overwriting with `{v}`')


    def __call__(self):
        if self.method == 'l2prune':
            prune = L2prune(self)
            prune()
            self.saveWeightsAndParams()
        if self.method == 'centripetalSGD':
            prune = L2prune(self)
            prune()
            self.saveWeightsAndParams()
        if self.method == 'geometricmedian':
            prune = GeometricMedian(self)
            prune()
            self.saveWeightsAndParams()
        else:
            'No valid method was chosen, exiting'
            quit()


    def saveWeightsAndParams(self):
        self.params.network.save(os.path.join(self.storage, "pruned.pt"))
        return


if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(
        description='Prune a given network for a given percentage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('percentage', help='Percentage of network to be pruned', type=float)
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-s', '--storage', metavar='folder', help='Storage folder', default='./pruned')
    parser.add_argument('-me', '--method', choices=['l2prune', 'centripetalSGD', 'geometricmedian'], default='l2prune',
                        help='The pruning method that will be used')
    parser.add_argument('-m', '--manner', choices=['hard', 'soft'], default='hard',
                        help='The manner in which to prune: soft or hard')
    args = parser.parse_args()

    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            print("CUDA enabled")
            device = torch.device('cuda')
        else:
            print("CUDA not available")

    if not os.path.isdir(args.storage):
        if not os.path.exists(args.storage):
            print('Pruning storage folder does not exist, creating...')
            os.makedirs(args.storage)
        else:
            raise ValueError('Storage path is not a folder')

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)

    # Start pruning
    prune = Pruning(
        params=params,
        device=device,
        storage=args.storage,
        method=args.method,
        percentage=args.percentage,
        manner=args.manner,
    )
    prune()
