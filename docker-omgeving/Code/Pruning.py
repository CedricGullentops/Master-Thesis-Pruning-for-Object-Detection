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
from change import getnet, getlossfunction
from L2prune import L2prune

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages
bb.logger.setConsoleLevel('ERROR')  # Only show error log messages


class Pruning:
    def __init__(self, model, method, loss, percentage, manner,  **kwargs):
        self.method = method
        self.model = model
        self.percentage = percentage
        self.manner = manner
        self.loss = lossfunction

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
        if self.method == 'centripetalSGD':
            prune = L2prune(self)
            prune()
        if self.method == 'geometricmedian':
            prune = L2prune(self)
            prune()
        else:
            'No valid method was chosen, exiting'
            quit()


if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(
        description='Prune a given network for a given percentage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('percentage', help='Percentage of network to be pruned', type=float)
    parser.add_argument('-ml', '--maxloss', help='Maximum acceptable loss to accuracy.', type=float)
    parser.add_argument('-me', '--method', choices=['l2prune', 'centripetalSGD', 'geometricmedian'], default='l2prune',
                        help='The pruning method that will be used')
    parser.add_argument('-l', '--loss', help='Loss function to use, default is RegionLoss.',
                        choices=['RegionLoss'], default='RegionLoss')
    parser.add_argument('-n', '--network', help='Pretrained network to prune',
                        choices=['Yolo', 'Yolt', 'DYolo', 'TinyYolo', 'MobileNetYolo'], default='Yolo')
    parser.add_argument('-m', '--manner', choices=['hard', 'soft'], default='hard',
                        help='The manner in which to prune: soft or hard')
    parser.add_argument('-o', '--optimizer', help='Optimizer to use')
    args = parser.parse_args()

    net = getnet(args.network)
    lossfunction = getlossfunction(args.loss, net)

    # Start pruning
    prune = Pruning(
        model=net,
        method=args.method,
        loss=lossfunction,
        percentage=args.percentage,
        manner=args.manner,
        optimizer=args.optimizer,
        maxloss=args.maxloss,
    )
    prune()
