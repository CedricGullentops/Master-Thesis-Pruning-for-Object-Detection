#
#   Main pruning class
#   Simple running example: python3 Pruning.py 10.0 backup/yolov2-640/final.pt -n cfg/yolov2-640.py -c
#

# Basic imports
import torch
import numpy as np
import argparse
import brambox as bb
import lightnet as ln
from L2prune import L2prune
from GeometricMedian import GeometricMedian
from utils import makeDependencyList
from testengine import TestEngine
from trainengine import TrainEngine
import os
import logging
from dataset import *
import multiprocessing as mp

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages
bb.logger.setConsoleLevel('ERROR')  # Only show error log messages

logprune = logging.getLogger('lightnet.FLIR.prune')

class Pruning:
    def __init__(self, testing_dataloader, training_dataloader, params, device, percentage, method, manner, storage, backup, maxloss, **kwargs):
        self.testing_dataloader = testing_dataloader
        self.training_dataloader = training_dataloader
        self.params = params
        self.device = device
        self.percentage = percentage
        self.method = method
        self.manner = manner
        self.storage = storage
        self.backup = backup
        self.maxloss = maxloss
        self.dependencies = makeDependencyList(self.params.network)

        # Setting kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                logprune.error('{k} attribute already exists, not overwriting with `{v}`')


    def __call__(self):
        prunecount = 0
        traineng = self.makeTrainEngine()
        testeng = self.makeTestEngine()
        self.params.network.eval()
        #current_accuracy = testeng()
        #original_accuracy = current_accuracy
        original_accuracy = 100.0
        current_accuracy = 100.0
        logstring = "Original accuracy is " + str(original_accuracy)
        logprune.info(logstring)
        maxiter = 5
        while current_accuracy >= (original_accuracy - self.maxloss):
            logprune.info("Pruning network")
            if self.method == 'l2prune':
                prune = L2prune(self, logprune)
                prune()
            elif self.method == 'centripetalSGD':
                #TODO
                quit()
            elif self.method == 'geometricmedian':
                prune = GeometricMedian(self, logprune)
                prune()
            else:
                logprune.critical('No valid method was chosen, exiting.')
                quit()()
            prunecount += 1
            for i in range(maxiter):
                self.params.network.train()
                self.params.optimizer = torch.optim.SGD(
                    params.network.parameters(),
                    lr = .001,
                    momentum = .9,
                    weight_decay = .0005,
                    dampening = 0,
                )
                self.params.optimizer.zero_grad()
                traineng()
                traineng.batch = 0
                self.params.network.eval()
                current_accuracy = testeng()
                logstring = "Current accuracy is " + str(current_accuracy)
                logprune.info(logstring)
                if (current_accuracy > (original_accuracy - self.maxloss)):
                    self.saveWeightsAndParams(prunecount, True)
                    logprune.info("Current accuracy is sufficient to continue pruning")
                    break
                else:
                    if i == maxiter-1:
                        logprune.warning("Couldn't reach original accuracy, saving and exiting")
                        self.saveWeightsAndParams(prunecount, False)
                        break


    def saveWeightsAndParams(self, prunecount, succesful):
        if succesful:
            self.params.network.save(os.path.join(self.storage, f"{self.manner}_pruned_{prunecount*self.percentage}.pt"))
        else:
            self.params.network.save(os.path.join(self.storage, f"{self.manner}_pruned_{prunecount*self.percentage}_FAILED.pt"))
        return


    def makeTestEngine(self):
        # Start test
        eng = TestEngine(
            self.params, self.testing_dataloader,
            device=self.device,
            loss_format=self.loss_format,
            detection=self.detection,
        )
        return eng

    
    def makeTrainEngine(self):
        eng = TrainEngine(
            self.params, self.training_dataloader,
            device=self.device,
            # visdom=self.visdom, plot_rate=self.visdom_rate, 
            backup_folder=self.backup
        )
        return eng


if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(
        description='Prune a given network for a given percentage',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('percentage', help='Percentage of network to be pruned per iteration', type=float)
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('-n', '--network', help='Network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-s', '--storage', metavar='folder', help='Storage folder for pruned weights', default='./backup/pruned')
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder for whilst training', default='./backup')
    parser.add_argument('-me', '--method', choices=['l2prune', 'centripetalSGD', 'geometricmedian'], default='l2prune',
                        help='The pruning method that will be used')
    parser.add_argument('-m', '--manner', choices=['hard', 'soft'], default='hard',
                        help='The manner in which to prune: soft or hard')
    parser.add_argument('-mb', '--batches', type=float, help='The maximum amount of batches to to train.')
    parser.add_argument('-ml', '--maxloss', type=float, help='The maximum percentage of accuracy allowed to lose.', default=0)

    parser.add_argument('-l', '--loss', help='How to display loss', choices=['abs', 'percent', 'none'], default='abs')
    parser.add_argument('-d', '--det', help='Detection pandas file', default=None)

    args = parser.parse_args()

    logging.basicConfig(filename='file.log', filemode='w')
    filehandler = ln.logger.setLogFile('file.log', levels=('TRAIN', 'TEST', 'PRUNE'), filemode='w')
    ln.logger.setConsoleColor(False)
    ln.logger.setConsoleLevel(logging.NOTSET)

    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            logprune.debug("CUDA enabled")
            device = torch.device('cuda')
        else:
            logprune.error("CUDA not available")

    if not os.path.isdir(args.storage):
        if not os.path.exists(args.storage):
            logprune.warning('Pruning storage folder does not exist, creating...')
            os.makedirs(args.storage)
        else:
            raise ValueError('Storage path is not a folder')

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)

    if args.batches != None:
        params.max_batches = args.batches
        

    # Dataloaders
    testing_dataloader = torch.utils.data.DataLoader(
        FLIRDataset(params.test_set, params, False),
        batch_size = params.mini_batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    training_dataloader = ln.data.DataLoader(
        FLIRDataset(params.train_set, params, False),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Start pruning
    prune = Pruning(
        testing_dataloader, training_dataloader, params, device,
        percentage=args.percentage,
        method=args.method,
        manner=args.manner,
        storage=args.storage,
        backup=args.backup,
        maxloss=args.maxloss,
        detection=args.det,
        loss_format=args.loss,
    )
    prune()
