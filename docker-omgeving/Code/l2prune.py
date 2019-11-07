#
#   L2norm pruning
#   Credits: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#   https://github.com/zepx/pytorch-weight-prune
#

# Basic imports
import lightnet as ln
import numpy as np
from utils import arg_nonzero_min, printProgressBar

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class L2prune:
    def __init__(self, Pruning):
        self.Pruning = Pruning
        self.totalfilters = 0
        self.prunedfilters = 0
        for x in Pruning.convolutionlayerlist:
            self.totalfilters += Pruning.model.layers[x[0]][x[1]].out_channels
        self.pruneamount = 0.01*Pruning.percentage*self.totalfilters
        print('There are', self.totalfilters , 'filters in convolutional layers, attempting to prune', int(self.pruneamount), 'filters')
        
    def __call__(self):
        self.values = []
        for parameter in self.Pruning.model.parameters():
            if len(parameter.data.size()) == 4: # Select convolution layers
                parameter_np = parameter.data.cpu().numpy()
                values_this_layer = np.square(parameter_np).sum(axis=1).sum(axis=1).sum(axis=1)\
                    /(parameter_np.shape[1]*parameter_np.shape[2]*parameter_np.shape[3]) # Sum of squared values divided by the amount of values
            
                # normalization (important)
                values_this_layer = values_this_layer / np.sqrt(np.square(values_this_layer).sum())
                min_value, min_ind = arg_nonzero_min(list(values_this_layer))
                self.values.append([min_value, min_ind])
                #print('min value:',min_value,'min ind', min_ind)
        self.values = np.array(self.values)
        layer_index = np.argmin(self.values[:, 0])
        filter_index = int(self.values[layer_index, 1])

        #count = 0
        #for x in self.Pruning.convolutionlayerlist:
        #    print('convolutielaag', count, 'heeft', self.Pruning.model.layers[x[0]][x[1]].out_channels, 'filters')
        #    count +=1

        self.pruneFilter(layer_index, filter_index)
        self.prunedfilters += 1

        while self.prunedfilters < 20: #self.pruneamount:
            printProgressBar(self.prunedfilters, self.pruneamount)
            layer_index, filter_index = self.findFilter()
            self.pruneFilter(layer_index, filter_index)
            self.prunedfilters += 1   


    def findFilter(self):
        self.values = []
        for parameter in self.Pruning.model.parameters():
            if len(parameter.data.size()) == 4: # Select convolution layers
                parameter_np = parameter.data.cpu().numpy()
                values_this_layer = np.square(parameter_np).sum(axis=1).sum(axis=1).sum(axis=1)\
                    /(parameter_np.shape[1]*parameter_np.shape[2]*parameter_np.shape[3]) # Sum of squared values divided by the amount of values
            
                # normalization (important)
                values_this_layer = values_this_layer / np.sqrt(np.square(values_this_layer).sum())
                min_value, min_ind = arg_nonzero_min(list(values_this_layer))
                self.values.append([min_value, min_ind])
                #print('min value:',min_value,'min ind', min_ind)
        self.values = np.array(self.values)
        layer_index = np.argmin(self.values[:, 0])
        filter_index = int(self.values[layer_index, 1])

        return layer_index, filter_index


    def pruneFilter(self, layer_index, filter_index):
        print('Pruning filter',filter_index, '@ layer', layer_index)
        layer = 0
        for parameter in self.Pruning.model.parameters():
            if len(parameter.data.size()) == 4:
                if layer != layer_index:
                    layer += 1
                    continue
                parameter_np = parameter.data.cpu().numpy()
                print(parameter_np.shape)
                parameter_np = np.delete(parameter_np, filter_index, axis=0)
                print(parameter_np.shape)
                #Set weights
                break
        return
