#
#   Utility functions
#   

# Basic imports
import lightnet as ln
import torch
from change import isconvoltionlayer
import numpy as np

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


# Iterate through a list and prune the given filters
# ATTENTION: the order of given filters matters
def hardPruneFilters(model, prunelist):
    for filter in prunelist:
        print('Hard pruning filter', filter[1], '@ layer', filter[0])
        layer = 0
        for m in model.modules():
            if isconvoltionlayer(m):
                if layer != filter[0]:
                    layer += 1
                    continue
                m.weight.data = torch.cat((m.weight.data[:filter[1]], m.weight.data[filter[1]+1:]))
                m.out_channels -= 1
                break
        return


# Iterate through a list and prune the given filters
# ATTENTION: the order of given filters matters
def softPruneFilters(model, prunelist):
    for filter in prunelist:
        print('Soft pruning filter', filter[1], '@ layer', filter[0])
        layer = 0
        for m in model.modules():
            if isconvoltionlayer(m):
                if layer != filter[0]:
                    layer += 1
                    continue
                zeros = torch.zeros([1,m.weight.data.shape[1],m.weight.data.shape[2], m.weight.data.shape[3]])
                tussenstap = torch.cat((m.weight.data[:filter[1]], zeros))
                m.weight.data = torch.cat((tussenstap, m.weight.data[filter[1]+1:]))
                break
        return


# Find lowest non-zero value in an array
def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
