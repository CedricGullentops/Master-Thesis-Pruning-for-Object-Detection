#!/usr/bin/env python
import os
import logging
import time
import argparse
from math import isinf, isnan
from statistics import mean
import torch
import visdom
import numpy as np
import lightnet as ln
from dataset import *
from tqdm import tqdm
import pandas as pd
import brambox as bb

log = logging.getLogger('lightnet.FLIR.train')


class TrainEngine(ln.engine.Engine):
    def start(self):
        self.training_success = False
        self.accuracy = None
        self.params.to(self.device)
        self.resize()
        self.optimizer.zero_grad()

        self.train_loss = {'tot': [], 'coord': [], 'conf': []}
        #self.plot_train_loss = ln.engine.LinePlotter(self.visdom, 'train_loss', opts=dict(xlabel='Batch', ylabel='Loss', title='Training Loss', showlegend=True, legend=['Total loss', 'Coordinate loss', 'Confidence loss']))
        #self.plot_lr = ln.engine.LinePlotter(self.visdom, 'learning_rate', name='Learning Rate', opts=dict(xlabel='Batch', ylabel='Learning Rate', title='Learning Rate Schedule'))
        #self.batch_end(self.plot_rate)(self.plot)

    def process_batch(self, data):
        data, target = data
        data = data.to(self.device)

        out = self.network(data)
        loss = self.loss(out, target) / self.batch_subdivisions
        loss.backward()

        self.train_loss['tot'].append(self.loss.loss_tot.item())
        self.train_loss['coord'].append(self.loss.loss_coord.item())
        self.train_loss['conf'].append(self.loss.loss_conf.item())

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step(self.batch, epoch=self.batch)

        # Get values from last batch
        tot = mean(self.train_loss['tot'][-self.batch_subdivisions:])
        coord = mean(self.train_loss['coord'][-self.batch_subdivisions:])
        conf = mean(self.train_loss['conf'][-self.batch_subdivisions:])
        self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f})')

        if isinf(tot) or isnan(tot):
            log.error('Infinite loss')
            self.sigint = True
            return

    def plot(self):
        tot = mean(self.train_loss['tot'])
        coord = mean(self.train_loss['coord'])
        conf = mean(self.train_loss['conf'])
        self.train_loss = {'tot': [], 'coord': [], 'conf': []}

        #self.plot_train_loss(np.array([[tot, coord, conf]]), np.array([self.batch]))
        #self.plot_lr(np.array([self.optimizer.param_groups[0]['lr']]), np.array([self.batch]))

    @ln.engine.Engine.batch_end(10)
    def resize(self):
        if self.batch >= self.max_batches - 200:
        	self.dataloader.change_input_dim(self.input_dimension, None)
        else:
        	self.dataloader.change_input_dim(self.resize_factor, self.resize_range)

    @ln.engine.Engine.epoch_end()
    def test_accuracy(self):
        self.network.eval()

        # Run network
        anno, det = [], []
        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(self.test_dataloader, desc='Test')):
                data = data.to(self.device)
                output = self.network(data)
                output = self.post(output)

                output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
                anno.append(target)
                det.append(output)

        anno = bb.util.concat(anno, ignore_index=True, sort=False)
        det = bb.util.concat(det, ignore_index=True, sort=False)
        self.network.train()
        
        # Statistics
        aps = []
        for c in tqdm(self.class_label_map, desc='Stat'):
            anno_c = anno[anno.class_label == c]
            det_c = det[det.class_label == c]

            # By default brambox considers ignored annos as regions -> we want to consider them as annos still
            matched_det = bb.stat.match_det(det_c, anno_c, 0.5, criteria=bb.stat.coordinates.iou, ignore=bb.stat.IgnoreMethod.SINGLE)
            pr = bb.stat.pr(matched_det, anno_c)

            aps.append(bb.stat.ap(pr))
        self.accuracy = round(100 * mean(aps), 2)
        log.info(f'[Pruned {self.prune_percentage}] Accuracy after training epoch {self.epoch}: {self.accuracy:.2f}%')

        # Check mAP
        if self.accuracy >= self.original_acc + self.upper_acc_delta:
            self.training_success = True

    def quit(self):
        if self.training_success:
            self.params.network.save(os.path.join(self.backup_folder, f"pruned_{self.prune_percentage}.pt"))
            return True
        elif self.batch >= self.max_batches:
            self.test_accuracy()
            if self.accuracy >= self.original_acc + self.lower_acc_delta:
                self.training_success = True
                self.params.network.save(os.path.join(self.backup_folder, f"pruned_{self.prune_percentage}.pt"))
            else:
                self.training_success = False
                self.params.network.save(os.path.join(self.backup_folder, f"pruned_{self.prune_percentage}-FAILED.pt"))
            return True
        elif self.sigint:
            self.params.save(os.path.join(self.backup_folder, 'backup.state.pt'))
            self.training_success = False
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('weight', help='Path to weight file', default=None, nargs='?')
    parser.add_argument('-n', '--network', help='network config file', required=True)
    parser.add_argument('-pn', '--prunednetwork', help='pruned network config file', required=True)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder', default='./backup')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('-e', '--visdom_env', help='Visdom environment to plot to', default='main')
    parser.add_argument('-p', '--visdom_port', help='Port of the visdom server', type=int, default=8080)
    parser.add_argument('-r', '--visdom_rate', help='How often to plot to visdom (batches)', type=int, default=1)
    args = parser.parse_args()

    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            log.warning('Backup folder does not exist, creating...')
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
            params.network.load(args.weight, strict=False)  # Disable strict mode for loading partial weights

    # Dataloader
    training_loader = ln.data.DataLoader(
        FLIRDataset(params.train_set, params, False),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.brambox_collate,
    )

    # Start training
    eng = TrainEngine(
        params, training_loader,
        device=device, visdom=visdom, plot_rate=args.visdom_rate, backup_folder=args.backup
    )
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch
    log.info(f'Training {b2-b1} batches took {t2-t1:.2f} seconds [{(t2-t1)/(b2-b1):.3f} sec/batch]')
