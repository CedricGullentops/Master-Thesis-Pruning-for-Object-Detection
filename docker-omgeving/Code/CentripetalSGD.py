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
import collections
import copy
from centripetalSGDTrainingEngine import CentripetalSGDTrainEngine


# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class CentripetalSGD():
    def __init__(self, Pruning, logprune, clustermethod):
        self.Pruning = Pruning
        self.logprune = logprune
        self.prunedfilters = 0
        self.clustermethod = clustermethod
       
        
    def __call__(self):
        self.values = []
        logstring = ""

        filtersperlayer = []
        filtersperlayerpruned = []

        layer = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                filtersperlayer.append(m.out_channels)
                layer +=1

        clusterlist = []
        for number in range(layer):
            if (self.Pruning.dependencies[number][2] == True):
                logstring = "working in layer " + str(number)
                self.logprune.info(logstring)
                clusterlist.append(self.clusterInLayer(number))
        centripetalTrainEngine = self.makeCentripetalSGDTrainEngine(clusterlist)
        centripetalTrainEngine()

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
        quit()

    def makeCentripetalSGDTrainEngine(self, clusterlist):
        eng = CentripetalSGDTrainEngine(
            self.Pruning.params, self.Pruning.training_dataloader,
            device=self.Pruning.device,
            # visdom=self.visdom, plot_rate=self.visdom_rate, 
            backup_folder=self.Pruning.backup,
            clusterlist=clusterlist
        )
        return eng

    def clusterInLayer(self, layer):
        count = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                if count != layer:
                    count += 1
                    continue
                clusterAmount = int(((100.0-self.Pruning.percentage) * m.out_channels) / 100.0)
                return self.makeClusters(m, clusterAmount)


    def centripetalSGDTraining(self, lr, weight_decay, centripetal, clusters, m):
        for cluster in clusters:
            for iteration in range(10000):
                deltaFilters = [torch.zeros([m.weight.data.shape[1],m.weight.data.shape[2],m.weight.data.shape[3]], device=self.Pruning.device)] * len(cluster)
                print(deltaFilters)

                for deltaFilter in deltaFilters:
                    lossIncrement = [torch.zeros([m.weight.data.shape[1],m.weight.data.shape[2],m.weight.data.shape[3]], device=self.Pruning.device)] #Eerste term
                print(deltaFilters)

                dindex = 0
                for dfilter in deltaFilters:
                    m.weight.data[cluster[dindex]] += lr * dfilter 
            quit()
            #TODO combine in_channels and delete other filters and in channels
        return


    def makeClusters(self, m, clusterAmount):
        clusters = []
        for _ in range(clusterAmount):
            clusters.append([])
        if (self.clustermethod == 'even'):
            count = 0
            for filter in range(m.out_channels):
                clusters[count].append(filter)
                count += 1
                if count == clusterAmount:
                    count = 0
            logstring = "Clusters: " + str(clusters)
            self.logprune.info(logstring)
            return clusters
        elif (self.clustermethod == 'kmeans'):
            centroids = []
            for i in range(clusterAmount):
                centroids.append(m.weight.data[i].clone().detach())
            clusters = self.kmeansClustering(m, clusters, centroids)
            logstring = "Clusters: " + str(clusters)
            self.logprune.info(logstring)
            return clusters
        else:
            self.logprune.critical('The given clusteringmethod wasn\'t recognized, exiting.')
            quit()


    def kmeansClustering(self, m, clusters, centroids):
        changedCluster = True
        oldClusters = [[None]] * len(clusters)
        while changedCluster == True:
            changedCluster = False
            for cluster in clusters:
                cluster.clear()
            for filter in range(m.out_channels):
                closestCentroidIndex = self.findClosesCentroid(m, filter, centroids)
                clusters[closestCentroidIndex].append(filter)
            clustercount = 0
            for cluster in clusters:
                oldclustercount = 0
                for oldCluster in oldClusters:
                    if oldclustercount == clustercount:
                        for filter in cluster:
                            if filter not in oldCluster:
                                changedCluster = True
                    oldclustercount += 1
                clustercount += 1
            if changedCluster == True:
                oldClusters = copy.copy(clusters)
                centroidCount = 0
                for centroid in centroids:
                    filterlist = []
                    if not clusters[centroidCount]:
                        continue
                    for filter in clusters[centroidCount]:
                        filterlist.append(m.weight.data[filter])
                    centroid = torch.mean(torch.stack(filterlist), 0)
                    centroidCount += 1
        
        # WARNING: This will delete more filters then demanded
        for i in range(len(clusters)-1,0,-1):
            if not clusters[i]:
                clusters.remove(clusters[i])
        return clusters


    def findClosesCentroid(self, m, filter, centroids):
        closestIndex = 0
        closestDistance = 100000
        centroidCount = 0
        for centroid in centroids:
            difference = m.weight.data[filter] - centroid
            distance = (difference.pow(2).sum(1).sum(1).sum(0) / difference.shape[0]*difference.shape[1]*difference.shape[2]).item()
            if closestDistance > distance:
                closestIndex = centroidCount
                closestDistance = distance
            centroidCount += 1
        return closestIndex
