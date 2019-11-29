#
#   Some random tests
#

# Basic imports
import subprocess
import lightnet as ln
import torch
import os

test = 'cfg/cfg.py'
params = ln.engine.HyperParameters.from_file(test)
weight = 'pruned/pruned.pt'

params.network.load(weight, strict=False)  # Disable strict mode for loading partial weights
