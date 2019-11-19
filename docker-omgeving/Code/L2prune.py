#
#   L2norm pruning
#

# Basic imports
import lightnet as ln
import torch
import numpy as np
from utils import arg_nonzero_min, printProgressBar, hardPruneFilters, softPruneFilters
from change import isconvoltionlayer

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class L2prune:
    def __init__(self, Pruning):
        self.Pruning = Pruning
        self.totalfilters = 0
        self.prunedfilters = 0
       
        
    def __call__(self):
        self.values = []
        if (self.Pruning.manner == 'soft'):
            self.modelcopy = self.Pruning.model
            self.prunelist = []

        # Calculate the initial norm values and
        # calculate total amount of filters and the amount to prune
        for m in self.Pruning.model.modules():
            if isconvoltionlayer(m):
                self.totalfilters += m.out_channels
                with torch.no_grad():
                    # Sum of squared values divided by the amount of values
                    weights = m.weight.data
                    values_this_layer = weights.pow(2).sum(1).sum(1).sum(1) / weights.shape[1]*weights.shape[2]*weights.shape[3]
                    # normalization (important)
                    values_this_layer = values_this_layer / torch.sqrt(torch.pow(values_this_layer, 2).sum())
                    min_value, min_ind = arg_nonzero_min(list(values_this_layer))
                    self.values.append([min_value, min_ind])
        self.values = np.array(self.values)
        self.pruneamount = int(0.01*self.Pruning.percentage*self.totalfilters)
        print('There are', self.totalfilters , 'filters in convolutional layers, attempting to prune', self.pruneamount, 'filters')

        # Prune as long as the conditions aren't met
        while self.prunedfilters < self.pruneamount:
            printProgressBar(self.prunedfilters, self.pruneamount)

            layer_index = self.findLayerIndex()
            filter_index = int(self.values[layer_index, 1])

            prunetuple = (layer_index, filter_index)
            prunelist = [prunetuple]

            if self.Pruning.manner == 'soft':
                self.prunelist.append(prunetuple) 

            hardPruneFilters(self.Pruning, prunelist)
            self.prunedfilters += 1 
            self.updateNormValues(layer_index)   

        if self.Pruning.manner == 'soft':
                softPruneFilters(self.modelcopy, self.prunelist)
                self.Pruning.model = self.modelcopy

        # check final amount of filters
        finalcount = 0
        for m in self.Pruning.model.modules():
            if isconvoltionlayer(m):
                finalcount += m.out_channels
        print ("The final amount of filters after pruning is", finalcount)
        print (self.Pruning.manner ,"pruned", self.prunedfilters,"filters")


    # Find the layer with the lowest value that is able to be pruned
    def findLayerIndex(self):
        smallestvalue = np.argmax(self.values[:, 1]) # Initiate with highest number
        smallestindex = 0
        count = 0
        for layer in self.values:
            if layer[0] < smallestvalue and self.Pruning.dependencies[count][2] == True:
                smallestvalue = layer[0]
                smallestindex = count
            count += 1
        return smallestindex


    # Update the norm values in self.values after pruning
    def updateNormValues(self, layer_index):
        count = 0
        for m in self.Pruning.model.modules():
            if isconvoltionlayer(m):
                if (count == layer_index):
                    with torch.no_grad():
                        # Sum of squared values divided by the amount of values
                        weights = m.weight.data
                        values_this_layer = weights.pow(2).sum(1).sum(1).sum(1) / weights.shape[1]*weights.shape[2]*weights.shape[3]
                        # normalization (important)
                        values_this_layer = values_this_layer / torch.sqrt(torch.pow(values_this_layer, 2).sum())
                        min_value, min_ind = arg_nonzero_min(list(values_this_layer))
                        self.values[layer_index] = [min_value, min_ind]
                        break
                else:
                    count += 1
