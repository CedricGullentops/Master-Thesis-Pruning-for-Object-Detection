#
#   L2norm pruning
#

# Basic imports
import lightnet as ln
import torch
import numpy as np
from utils import arg_nonzero_min, hardPruneFilters, softPruneFilters
from utils import isConvolutionLayer
import logging

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class L2prune:
    def __init__(self, Pruning, logprune):
        self.Pruning = Pruning
        self.logprune = logprune
        self.totalfilters = 0
        self.prunedfilters = 0
       
        
    def __call__(self):
        with torch.no_grad():
            self.values = []
            if (self.Pruning.manner == 'soft'):
                self.modelcopy = self.Pruning.params.network
                self.prunelist = []

            filtersperlayer = []
            filtersperlayerpruned = []
            logstring = ""

            # Calculate the initial norm values and
            # calculate total amount of filters and the amount to prune
            for m in self.Pruning.params.network.modules():
                print(m)
                if isConvolutionLayer(m):
                    filtersperlayer.append(m.out_channels)
                    self.totalfilters += m.out_channels
                    
                    # Sum of squared values divided by the amount of values
                    weights = m.weight.data
                    values_this_layer = weights.pow(2).sum(1).sum(1).sum(1) / weights.shape[1]*weights.shape[2]*weights.shape[3]
                    # normalization (important)
                    values_this_layer = values_this_layer / torch.sqrt(torch.pow(values_this_layer, 2).sum())
                    min_value, min_ind = arg_nonzero_min(list(values_this_layer))
                    self.values.append([min_value, min_ind])
            self.values = np.array(self.values)
            self.pruneamount = int(0.01*self.Pruning.percentage*self.totalfilters)
            logstring = 'There are ' + str(self.totalfilters)  + ' filters in convolutional layers, attempting to prune ' + str(self.pruneamount) + ' filters'
            self.logprune.info(logstring)

            # Prune as long as the conditions aren't met
            while self.prunedfilters < self.pruneamount:
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
                    softPruneFilters(self.Pruning, self.modelcopy, self.prunelist)
                    self.Pruning.params.network = self.modelcopy

            # check final amount of filters
            finalcount = 0
            for m in self.Pruning.params.network.modules():
                if isConvolutionLayer(m):
                    filtersperlayerpruned.append(m.out_channels)
                    finalcount += m.out_channels
            logstring = "The final amount of filters after pruning is " + str(finalcount)
            self.logprune.info(logstring)
            logstring = str(self.Pruning.manner) + " pruned " + str(self.prunedfilters) + " filters"
            self.logprune.info(logstring)
            self.logprune.info("Filters before pruning:")
            self.logprune.info(filtersperlayer)
            self.logprune.info("Filters after pruning:")
            self.logprune.info(filtersperlayerpruned)


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
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
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
