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
from CentripetalSGD import CentripetalSGD
from utils import makeDependencyList
from testengine import TestEngine
from trainengine import TrainEngine
import os
import logging
from dataset import *
import multiprocessing as mp
import pandas as pd
import brambox as bb
from tqdm import tqdm
from statistics import mean

# Settings
bb.logger.setConsoleLevel('ERROR')  # Only show error log messages
ln.logger.setConsoleLevel('INFO')
logprune = logging.getLogger('lightnet.FLIR.prune')
torch.set_num_threads(8)    # Specify threads for DGX

class Pruning:
    def __init__(self, testing_dataloader, training_dataloader, params, device, percentage, method, manner, storage, **kwargs):
        self.testing_dataloader = testing_dataloader
        self.training_dataloader = training_dataloader
        self.params = params
        self.device = device
        self.percentage = percentage
        self.method = method
        self.manner = manner
        self.storage = storage
        self.dependencies = makeDependencyList(self.params.network)

        # Setting kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                logprune.error('{k} attribute already exists, not overwriting with `{v}`')

    def __call__(self):
        self.params.network.to(self.device)
        prunecount = 0
        current_accuracy = self.test_accuracy()
        original_accuracy = current_accuracy
        logprune.info(f'Original accuracy is {original_accuracy:.2f}%')
        while current_accuracy >= (original_accuracy + self.params.lower_acc_delta):
            # Prune
            logprune.info("Pruning network")
            if self.method == 'l2prune':
                prune = L2prune(self, logprune)
                prune()
            elif self.method == 'centripetalSGD_even':
                prune = CentripetalSGD(self, logprune, 'even')
                prune()
            elif self.method == 'centripetalSGD_kmeans':
                prune = CentripetalSGD(self, logprune, 'kmeans')
                prune()
            elif self.method == 'geometricmedian':
                prune = GeometricMedian(self, logprune)
                prune()
            else:
                logprune.critical('No valid method was chosen, exiting.')
                quit()
            prunecount += 1
            self.params.network.to('cpu')
            self.params.network.to(self.device)

            # Check accuracy
            current_accuracy = self.test_accuracy()
            logprune.info(f'Accuracy after pruning: {current_accuracy:.2f}%')

            #import pdb; pdb.set_trace()

            # Train
            self.params.network.train()
            self.params.batch = 0
            self.params.epoch = 0
            self.params.optimizer = torch.optim.SGD(
                params.network.parameters(),
                lr = self.params.lr,
                momentum = self.params.momentum,
                weight_decay = self.params.weight_decay,
                dampening = self.params.dampening,
            )
            burn_in = torch.optim.lr_scheduler.LambdaLR(
                self.params.optimizer,
                lambda b: (b / self.params.burnin) ** 4,
            )
            step = torch.optim.lr_scheduler.MultiStepLR(
                params.optimizer,
                milestones = self.params.milestones,
                gamma = self.params.gamma,
            )
            params.scheduler = ln.engine.SchedulerCompositor(
                (0,                     burn_in),
                (self.params.burnin,    step),
            )

            training_engine = self.makeTrainEngine(original_accuracy, prunecount*self.percentage)
            training_engine()
            if training_engine.sigint:  # Interrupted
                break
            current_accuracy = training_engine.accuracy
            
            logprune.info(f'[Pruned {prunecount*self.percentage}] Trained for {training_engine.batch} batches')
            logprune.info(f'[Pruned {prunecount*self.percentage}] Accuracy after retraining: {current_accuracy:.2f}%')
            if not training_engine.training_success:
                logprune.warning(f"[Pruned {prunecount*self.percentage}] Couldn't reach original accuracy, saving and exiting")
                break

    def saveWeights(self, prunecount, succesful):
        if succesful:
            self.params.network.save(os.path.join(self.storage, f"{self.manner}_pruned_{prunecount*self.percentage}.pt"))
        else:
            self.params.network.save(os.path.join(self.storage, f"{self.manner}_pruned_{prunecount*self.percentage}_FAILED.pt"))
        return

    def test_accuracy(self):
        self.params.network.eval()

        # Run network
        anno, det = [], []
        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(self.testing_dataloader, desc='Test')):
                data = data.to(self.device)
                output = self.params.network(data)
                output = self.params.post(output)

                output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
                anno.append(target)
                det.append(output)

        anno = bb.util.concat(anno, ignore_index=True, sort=False)
        det = bb.util.concat(det, ignore_index=True, sort=False)
        self.params.network.train()
        
        # Statistics
        aps = []
        for c in tqdm(self.params.class_label_map, desc='Stat'):
            anno_c = anno[anno.class_label == c]
            det_c = det[det.class_label == c]

            # By default brambox considers ignored annos as regions -> we want to consider them as annos still
            matched_det = bb.stat.match_det(det_c, anno_c, 0.5, criteria=bb.stat.coordinates.iou, ignore=bb.stat.IgnoreMethod.SINGLE)
            pr = bb.stat.pr(matched_det, anno_c)

            aps.append(bb.stat.ap(pr))

        return 100 * mean(aps)

    def makeTrainEngine(self, original_acc, prune_percentage):
        return TrainEngine(
            self.params, self.training_dataloader,
            device=self.device,
            backup_folder=self.storage,
            test_dataloader=self.testing_dataloader,
            original_acc=original_acc,
            prune_percentage=prune_percentage,
            # visdom=self.visdom, plot_rate=self.visdom_rate, 
        )


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
    parser.add_argument('-me', '--method', choices=['l2prune', 'centripetalSGD_even', 'centripetalSGD_kmeans', 'geometricmedian'], default='l2prune',
                        help='The pruning method that will be used')
    parser.add_argument('-m', '--manner', choices=['hard', 'soft'], default='hard',
                        help='The manner in which to prune: soft or hard')
    parser.add_argument('-mb', '--batches', type=float, help='The maximum amount of batches to to train.')
    parser.add_argument('-ud', '--upperdelta', type=float, help='The maximum percentage of accuracy to gain before stopping training routine', default=None)
    parser.add_argument('-ld', '--lowerdelta', type=float, help='The maximum percentage of accuracy allowed to lose. [NEGATIVE]', default=None)
    args = parser.parse_args()

    #logging.basicConfig(filename='file.log', filemode='w')
    if not os.path.isdir(args.storage):
        if not os.path.exists(args.storage):
            logprune.warning('Pruning storage folder does not exist, creating...')
            os.makedirs(args.storage)
        else:
            raise ValueError('Storage path is not a folder')

    if args.storage is not None:
        logging.basicConfig(filename=os.path.join(args.storage, 'file.log'), filemode='w')
    else:
        logging.basicConfig(filename='file.log', filemode='w')

    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            logprune.debug("CUDA enabled")
            device = torch.device('cuda')
        else:
            logprune.error("CUDA not available")

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)

    if args.batches is not None:
        params.max_batches = args.batches
    if args.upperdelta is not None:
        params.upper_acc_delta = args.upperdelta
    if args.lowerdelta is not None:
        params.lower_acc_delta = args.lowerdelta
        
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
        testing_dataloader, training_dataloader, params, device,
        percentage=args.percentage,
        method=args.method,
        manner=args.manner,
        storage=args.storage,
    )
    prune()
