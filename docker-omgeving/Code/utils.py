#
#   Functions that are subject to change when changes are made to LightNet
#   Note that the argument parser will always have to be updated models or lossfunctions are added

# Basic imports
import lightnet as ln

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


# Returns a network 
def getnet(networkname):
    if networkname == 'Yolo':
        return ln.models.Yolo()
    elif networkname == 'Yolt':
        return ln.models.Yolt()
    elif networkname == 'DYolo':
        return ln.models.DYolo()
    elif networkname == 'TinyYolo':
        return ln.models.TinyYolo()
    elif networkname == 'MobileNetYolo':
        return ln.models.MobileNetYolo()
    else:
        print('An unsupported network was chosen, exiting.')
        quit()


# Returns an initialized loss function
def getlossfunction(lossfunctionname, model):
    if lossfunctionname == 'RegionLoss':
        loss = ln.network.loss.RegionLoss(
            num_classes=model.num_classes,
            anchors=model.anchors,
            stride=model.stride
        )
        return loss
    else:
        print('An unsupported lossfunction was chosen, exiting.')
        quit()


# Test if a layer is a convolution layer
def isconvoltionlayer(model, sequential, layer):
    if isinstance(model.layers[sequential][layer], ln.network.layer._darknet.Conv2dBatchReLU):
        return True
    elif isinstance(model.layers[sequential][layer], ln.network.layer._mobilenet.Conv2dDepthWise):
        return True
    else:
        return False


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
