import lightnet as ln
import torch
from PrunedNetwork import *

__all__ = ['params']


params = ln.engine.HyperParameters( 
    # Network
    class_label_map = ['person'],
    _input_dimension = (416, 416),
    _resize_range = (10, 19),
    _resize_factor = 32,
    _batch_size = 64,
    _mini_batch_size = 8,
    _max_batches = 20000,

    # Pruning
    _lower_acc_delta = -2,
    _upper_acc_delta = 1,

    # Dataset
    _train_set = 'data/sets/train.h5',
    _test_set = 'data/sets/test.h5',

    # Data Augmentation
    _jitter = .3,
    _flip = .5,
    _hue = 0,           # Original Grayscale images -> dont modify hue
    _saturation = 1.5,
    _value = 1.5,

    # Optimizer
    lr = .001,
    momentum = .9,
    weight_decay = .0005,
    dampening = 0,

    # Scheduler
    burnin = 1000,
    milestones = [15000],
    gamma = 0.1,
)

# Network
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

params.network = Prune(ln.models.Yolo(len(params.class_label_map)))
params.network.apply(init_weights)

# Loss
params.loss = ln.network.loss.RegionLoss(
    len(params.class_label_map),
    params.network.anchors,
    params.network.stride,
    coord_prefill=0,
)

# Postprocessing
params._post = ln.data.transform.Compose([
    ln.data.transform.GetBoundingBoxes(len(params.class_label_map), params.network.anchors, 0.001),
    ln.data.transform.NonMaxSuppression(0.5),
    ln.data.transform.TensorToBrambox(params.input_dimension, params.class_label_map),
])
