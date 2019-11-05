# Basic imports
import lightnet as ln
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import brambox as bb
import argparse

# Settings
ln.logger.setConsoleLevel('ERROR')             # Only show error log messages
bb.logger.setConsoleLevel('ERROR')             # Only show error log messages


class L2prune:
    def __init__(self, network, loss, percentage, **kwargs):
        self.network = network
        self.loss = loss
        self.percentage = percentage

        # Setting kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                log.error('{k} attribute already exists on TestEngine, not overwriting with `{v}`')	

    def __call__(self):
        print(self.network)
        return
        

    def prune(self):
        return


if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(
        description='Prune a given network for a given percentage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('loss', help='Allowed loss value', type=float)
    parser.add_argument('percentage', help='Percentage of network to be pruned', type=float)
    parser.add_argument('-n', '--network', help='Network to prune', choices=['Yolo', 'Yolt', 'DYolo', 'TinyYolo', 'MobileNetYolo'], default='Yolo')
    parser.add_argument('-o', '--optimizer', help='Optimizer to use')
    args = parser.parse_args()

    net = getnet(args.network)

    # Start pruning
    pruning = L2prune(
        net,
        loss=args.loss,
        percentage=args.percentage,
        optimizer=args.optimizer,
    )
    pruning()
