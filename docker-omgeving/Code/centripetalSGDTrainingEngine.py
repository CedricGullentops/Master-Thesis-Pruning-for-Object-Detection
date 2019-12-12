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

log = logging.getLogger('lightnet.FLIR.train')


class CentripetalSGDTrainEngine(ln.engine.Engine):
    def start(self):
        self.params.to(self.device)
        self.resize()
        self.optimizer.zero_grad()

        self.train_loss = {'tot': [], 'coord': [], 'conf': []}

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

    @ln.engine.Engine.batch_end(10)
    def resize(self):
        if self.batch >= self.max_batches - 200:
        	self.dataloader.change_input_dim(self.input_dimension, None)
        else:
        	self.dataloader.change_input_dim(self.resize_factor, self.resize_range)

    def quit(self):
        if self.batch >= self.max_batches:
            return True
        else:
             False
