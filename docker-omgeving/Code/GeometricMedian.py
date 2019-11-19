#
#   L2norm pruning
#

# Basic imports
import lightnet as ln
import torch
import numpy as np
from utils import arg_nonzero_min, printProgressBar, hardPruneFilters, softPruneFilters
from change import isConvolutionLayer
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class GeometricMedian():
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
            if isConvolutionLayer(m):
                self.totalfilters += m.out_channels
                layerAmount = int((self.Pruning.percentage * m.out_channels) / 100.0)
                
                with torch.no_grad():
                    centroid = self.findCentralPoint(m.weight.data)   
                    print("Centroid for layer", centroid)  

        # check final amount of filters
        finalcount = 0
        for m in self.Pruning.model.modules():
            if isConvolutionLayer(m):
                finalcount += m.out_channels
        print ("The final amount of filters after pruning is", finalcount)
        print (self.Pruning.manner ,"pruned", self.prunedfilters,"filters")


    def findCentralPoint(self, weights):
        weights = weights.numpy()
        centroid = np.zeros([weights.shape[1], weights.shape[2], weights.shape[3]]) # Initialiseer centerpunt

        def tensorToOneD(tensor, dim1, dim2, dim3):
            return tensor.reshape(dim1 * dim2 * dim3)

        carray = tensorToOneD(centroid, weights.shape[1], weights.shape[2], weights.shape[3])

        def oneDToArray(array, dim1, dim2, dim3):
            return array.reshape(dim1, dim2, dim3)

        def aggregate_distance(carray):
            centroid = oneDToArray(carray, weights.shape[1], weights.shape[2], weights.shape[3])
            filterdiff = weights - centroid
            return (np.square(filterdiff).sum(axis=1).sum(axis=1).sum(axis=1)/(filterdiff.shape[1]*filterdiff.shape[2]*filterdiff.shape[3])).sum()

        optimize_result = minimize(aggregate_distance, carray, method='COBYLA')

        return torch.from_numpy(optimize_result.x)
