#
#   GeometricMedian Pruning
#

# Basic imports
import lightnet as ln
import torch
import numpy as np
from utils import arg_nonzero_min, hardPruneFilters, softPruneFilters, deleteGrads
from utils import isConvolutionLayer
#import multiprocessing as mp
import logging


# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class CentripetalSGD():
    def __init__(self, Pruning, logprune, clustermethod):
        self.Pruning = Pruning
        self.logprune = logprune
        self.prunedfilters = 0
        self.clustermethod = clustermethod
       
        
    def __call__(self):
        with torch.no_grad():
            self.values = []
            logstring = ""

            filtersperlayer = []
            filtersperlayerpruned = []
            
            #pool = mp.Pool(mp.cpu_count()) # Allows parallelization

            layer = 0
            for m in self.Pruning.params.network.modules():
                if isConvolutionLayer(m):
                    filtersperlayer.append(m.out_channels)
                    layer +=1

            for number in range(layer):
                if (self.Pruning.dependencies[number][2] == True):
                    logstring = "working in layer " + str(number)
                    self.logprune.info(logstring)
                    #pool.apply(self.pruneInLayer, args=(number, )) 
                    self.pruneInLayer(number)
            #pool.close()

            # check final amount of filters
            finalcount = 0
            for m in self.Pruning.params.network.modules():
                if isConvolutionLayer(m):
                    filtersperlayerpruned.append(m.out_channels)
                    finalcount += m.out_channels

            deleteGrads(self.Pruning)

            logstring = "The final amount of filters after pruning is " +  str(finalcount)
            self.logprune.info(logstring)
            logstring = str(self.Pruning.manner) + " pruned " + str(self.prunedfilters) + " filters"
            self.logprune.info(logstring)
            self.logprune.info("Filters before pruning:")
            self.logprune.info(filtersperlayer)
            self.logprune.info("Filters after pruning:")
            self.logprune.info(filtersperlayerpruned)

    def pruneInLayer(self, layer):
        count = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                if count != layer:
                    count += 1
                    continue
                layerAmount = int((self.Pruning.percentage * m.out_channels) / 100.0)
                clusterAmount = m.out_channels - layerAmount
                self.makeClusters(m, clusterAmount)
                

    def makeClusters(self, m, clusterAmount):
        clusters = []
        for _ in range(clusterAmount):
            clusters.append([])
        if (self.clustermethod == 'even'):
            count = 0
            for filter in range(m.out_channels):
                clusters[count].append(filter)
                if count == clusterAmount-1:
                    count = 0
            print(clusters)
            return clusters
        elif (self.clustermethod == 'kmeans'):
            centroids = []
            for i in range(clusterAmount):
                centroids.append(m.weight.data[i].clone().detach())
            print(centroids)
            return self.kmeansClustering(m, clusters, centroids)
        else:
            self.logprune.critical('The given clusteringmethod wasn\'t recognized, exiting.')
            quit()

    def kmeansClustering(self, m, clusters, centroids):
        changedCluster = True
        while changedCluster == True:
            print('TEST')
        return clusters

