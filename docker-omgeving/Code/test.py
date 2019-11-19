#
#   Utility functions
#   

# Basic imports
import lightnet as ln
import torch
from utils import makeDependencyList

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages

model = ln.models.Yolo()
table = makeDependencyList(model)
for x in table:
    print(x)
