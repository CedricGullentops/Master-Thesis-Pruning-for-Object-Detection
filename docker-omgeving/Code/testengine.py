#
#   Functions that are subject to change when changes are made to LightNet
#   Note that the argument parser will always have to be updated models or lossfunctions are added

# Basic imports
import os
import argparse
import logging
from statistics import mean
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import lightnet as ln
import brambox as bb
from dataset import *

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages

class TestEngine:
    def __init__(self, params, dataloader, device, **kwargs):
        self.params = params
        self.dataloader = dataloader
        self.device = device

        # extract data from params
        self.post = params.post
        self.loss = params.loss
        self.network = params.network

        # Setting kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            else:
                log.error('{k} attribute already exists on TestEngine, not overwriting with `{v}`')

    def __call__(self):
        self.params.to(self.device)
        self.network.eval()
        self.loss.eval()    # This is necessary so the loss doesnt use its 'prefill' rule

        anno, det = self.test_none()

        aps = []
        for c in tqdm(self.params.class_label_map):
            anno_c = anno[anno.class_label == c]
            det_c = det[det.class_label == c]

            # By default brambox considers ignored annos as regions -> we want to consider them as annos still
            matched_det = bb.stat.match_det(det_c, anno_c, 0.5, criteria=bb.stat.coordinates.iou, ignore=bb.stat.IgnoreMethod.SINGLE)
            pr = bb.stat.pr(matched_det, anno_c)

            aps.append(bb.stat.ap(pr))

        m_ap = round(100 * mean(aps), 2)
        return m_ap


    def test_none(self):
        anno, det = [], []

        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(self.dataloader)):
                data = data.to(self.device)
                output = self.network(data)
                output = self.post(output)

                output.image = pd.Categorical.from_codes(output.image, dtype=target.image.dtype)
                anno.append(target)
                det.append(output)

        anno = bb.util.concat(anno, ignore_index=True, sort=False)
        det = bb.util.concat(det, ignore_index=True, sort=False)
        return anno, det
