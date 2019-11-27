#
#   L2norm pruning
#

# Basic imports
import lightnet as ln
import torch
import numpy as np
from utils import arg_nonzero_min, hardPruneFilters, softPruneFilters
from utils import isConvolutionLayer
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import multiprocessing as mp

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class GeometricMedian():
    def __init__(self, Pruning):
        self.Pruning = Pruning
        self.prunedfilters = 0
       
        
    def __call__(self):
        self.values = []
        if (self.Pruning.manner == 'soft'):
            self.modelcopy = self.Pruning.params.network
            self.prunelist = []

        filtersperlayer = []
        filtersperlayerpruned = []
        pool = mp.Pool(mp.cpu_count()) # Allows parallelization

        layer = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                filtersperlayer.append(m.out_channels)
                layer +=1

        for number in range(layer):
            pool.apply(self.pruneInLayer, args=(number, )) 
        pool.close()

        # check final amount of filters
        finalcount = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                filtersperlayerpruned.append(m.out_channels)
                finalcount += m.out_channels
        print ("The final amount of filters after pruning is", finalcount)
        print (self.Pruning.manner ,"pruned", self.prunedfilters,"filters")
        print("Filters before pruning:")
        print(filtersperlayer)
        print("Filters after pruning:")
        print(filtersperlayerpruned)


    def pruneInLayer(self, layer):
        count = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                if count != layer:
                    count += 1
                    continue
                layerAmount = int((self.Pruning.percentage * m.out_channels) / 100.0)
                with torch.no_grad():
                    centroid = self.findCentralPoint(m.weight.data)
                    order = self.arrangeFiltersByDistance(m.weight.data, centroid)
                    toDelete = []
                    for filter in range(layerAmount):
                        toDelete.append((layer, order[filter][0]))
                    toDelete = sorted(toDelete, key=lambda toDelete: toDelete[1], reverse=True)
                    print("Layer", layer, "deleting:", toDelete)

                    if self.Pruning.manner == 'soft':
                        softPruneFilters(self.Pruning.params.network, toDelete)
                    elif self.Pruning.manner == 'hard':
                        hardPruneFilters(self.Pruning, toDelete)
                    self.prunedfilters += len(toDelete)
                return


    def arrangeFiltersByDistance(self, weights, centroid):
        distance = []
        count = 0
        for filter in range(weights.shape[0]):
            difference = weights[filter] - centroid
            distance.append((count, difference.pow(2).sum(1).sum(1).sum(0) / difference.shape[0]*difference.shape[1]*difference.shape[2]))
            count += 1

        #for filter in distance:
        order = sorted(distance, key=lambda distance: distance[1])
        return order


    def findCentralPoint(self, weights):
        weights = weights.numpy()
        centroid = np.median(weights, 0) # Initial guess for centroid is median

        carray = centroid.ravel()

        def aggregate_distance(carray):
            centroid = carray.reshape(weights.shape[1], weights.shape[2], weights.shape[3])
            filterdiff = weights - centroid
            return (np.square(filterdiff).sum(axis=1).sum(axis=1).sum(axis=1)/(filterdiff.shape[1]*filterdiff.shape[2]*filterdiff.shape[3])).sum()

        optimize_result = minimize(aggregate_distance, carray, method='COBYLA')
        result = optimize_result.x.reshape(weights.shape[1], weights.shape[2], weights.shape[3])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor = torch.from_numpy(result).float().to(device)

        return tensor
