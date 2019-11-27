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
from dataset import *
from testengine import *
from trainengine import *

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages
bb.logger.setConsoleLevel('ERROR')  # Only show error log messages


class Pruning:
    def __init__(self, params, testing_dataloader, training_loader, device, method, percentage, manner,  **kwargs):
        self.method = method
        self.params = params
        self.testing_dataloader = testing_dataloader
        self.training_loader = training_loader
        self.device = device
        self.percentage = percentage
        self.manner = manner
        self.dependencies = makeDependencyList(self.params.network)
        self.visdom = visdom

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
            prune = GeometricMedian(self)
            prune()
        else:
            'No valid method was chosen, exiting'
            quit()


    def pruneLoop(self, prune):
        original_accuracy = self.getAccuracy()
        new_accuracy = 110
        while new_accuracy > original_accuracy:
            prune()
            new_accuracy = self.getAccuracy()
            
            eng = TrainEngine(
                self.params, self.training_loader,
                device=device, visdom=visdom, plot_rate=args.visdom_rate, backup_folder=args.backup
            )
            b1 = eng.batch
            t1 = time.time()
            eng()
            t2 = time.time()
            b2 = eng.batch
        print(f'Training {b2-b1} batches took {t2-t1:.2f} seconds [{(t2-t1)/(b2-b1):.3f} sec/batch]')
        return


    def getAccuracy(self):
        eng = TestEngine(
            self.params, self.testing_dataloader,
            device=self.device,
        )
        m_ap = eng()
        print(f'mAP: {m_ap:.2f}%')
        return m_ap


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
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder', default='./backup')
    parser.add_argument('-me', '--method', choices=['l2prune', 'centripetalSGD', 'geometricmedian'], default='l2prune',
                        help='The pruning method that will be used')
    parser.add_argument('-m', '--manner', choices=['hard', 'soft'], default='hard',
                        help='The manner in which to prune: soft or hard')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('-e', '--visdom_env', help='Visdom environment to plot to', default='main')
    parser.add_argument('-p', '--visdom_port', help='Port of the visdom server', type=int, default=8080)
    parser.add_argument('-r', '--visdom_rate', help='How often to plot to visdom (batches)', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            print("CUDA enabled")
            device = torch.device('cuda')
        else:
            print("CUDA not available")

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            print('Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            raise ValueError('Backup path is not a folder')

    if args.visdom:
        visdom = visdom.Visdom(port=args.visdom_port, env=args.visdom_env)
    else:
        visdom = None

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)

    # Dataloader
    testing_dataloader = torch.utils.data.DataLoader(
        FLIRDataset(params.test_set, params, False),
        batch_size = params.mini_batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Dataloader
    training_loader = ln.data.DataLoader(
        FLIRDataset(params.train_set, params, True),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Start pruning
    prune = Pruning(
        params = params,
        testing_dataloader = testing_dataloader,
        training_loader = training_loader,
        device=device,
        method=args.method,
        percentage=args.percentage,
        manner=args.manner,
    )
    prune()
