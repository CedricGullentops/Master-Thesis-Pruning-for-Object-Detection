#
#   Functions that are subject to change when changes are made to LightNet
#   Note that the argument parser will always have to be updated models or lossfunctions are added

# Basic imports
import lightnet as ln
import torch

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


# Returns a network 
def getNet(networkname):
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
def getLossFunction(lossfunctionname, model):
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
def isConvolutionLayer(module):
    if isinstance(module, torch.nn.Conv2d):
        return True
    else:
        return False
