#
#   GeometricMedian Pruning
#

# Basic imports
import lightnet as ln
import torch
import numpy as np
from utils import arg_nonzero_min, hardPruneFilters, softPruneFilters, deleteGrads, combineFilters, isConvolutionLayer
#import multiprocessing as mp
import logging
import collections
import copy
from C_SGD import C_SGD
from statistics import mean


# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


class CentripetalSGD():
    def __init__(self, Pruning, logprune, clustermethod, training_dataloader):
        self.Pruning = Pruning
        self.logprune = logprune
        self.prunedfilters = 0
        self.clustermethod = clustermethod
        self.training_dataloader = training_dataloader
        self.epoch = 0
        self.batch = 0
       
        
    def __call__(self):
        self.values = []
        logstring = ""

        filtersperlayer = []
        filtersperlayerpruned = []

        totalfilters = 0
        layer = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                filtersperlayer.append(m.out_channels)
                totalfilters += m.out_channels
                layer +=1

        # Generate clusters
        clusterlist = []
        for number in range(layer):
            if (self.Pruning.dependencies[number][2] == True):
                logstring = "Making clusters for layer " + str(number)
                self.logprune.info(logstring)
                clusterlist.append(self.clusterInLayer(number))

        self.Pruning.params.network.train()
        self.Pruning.params.batch = 0
        self.Pruning.params.epoch = 0
        self.Pruning.params.optimizer = C_SGD(
            self.Pruning.params.network.parameters(),
            clusterlist,
            self.Pruning,
            lr = 0.03,
            weight_decay = self.Pruning.params.weight_decay,
            centripetal_force = 0.003,
        )
        burn_in = torch.optim.lr_scheduler.LambdaLR(
            self.Pruning.params.optimizer,
            lambda b: (b / self.Pruning.params.burnin) ** 4,
        )
        step = torch.optim.lr_scheduler.MultiStepLR(
            self.Pruning.params.optimizer,
            milestones = self.Pruning.params.milestones,
            gamma = self.Pruning.params.gamma,
        )
        self.Pruning.params.scheduler = ln.engine.SchedulerCompositor(
            (0,                     burn_in),
            (self.Pruning.params.burnin,    step),
        )

        # TRAINER VAN HIER

        self.logprune.info('Start training')
        self.Pruning.params.network.train()

        batch_subdivisions = self.Pruning.params.batch_size // self.Pruning.params.mini_batch_size

        idx = 0
        while True:
            idx %= batch_subdivisions
            loader = self.training_dataloader
            for idx, data in enumerate(loader, idx+1):
                # Batch Start

                # Forward and backward on (mini-)batches
                # process_batch
                data, target = data
                data = data.to(self.Pruning.device)

                out = self.Pruning.params.network(data)
                loss = self.Pruning.params.loss(out, target) / batch_subdivisions
                loss.backward()

                if idx % batch_subdivisions != 0:
                    continue

                # Optimizer step
                self.batch += 1     # Should only be called after train, but this is easier to use self.batch in function
                logstring = "Cluster retraining batch: " + str(self.batch)
                self.logprune.info(logstring)

                #self.train_batch()
                self.Pruning.params.optimizer.step()
                self.Pruning.params.optimizer.zero_grad()
                self.Pruning.params.scheduler.step(self.batch, epoch=self.batch)

                # Batch End

            # Epoch End
            self.epoch += 1

            logstring = "Cluster retraining epoch: " + str(self.epoch)
            self.logprune.info(logstring)

            layer = 0
            allowedlayer = 0
            for m in self.Pruning.params.network.modules():
                if isConvolutionLayer(m):
                    if (self.Pruning.dependencies[layer][2] == True):
                        clustercount = -1
                        for cluster in clusterlist[allowedlayer]:
                            clustercount += 1
                            # Test to see if filters grew to eachoter
                            print("One cluster:")
                            filtercount = 0
                            for filter in cluster:
                                print(m.weight[filter])
                                filtercount += 1
                            break
                    break
            if self.epoch == 100:
                self.Pruning.params.optimizer = C_SGD(
                self.Pruning.params.network.parameters(),
                clusterlist,
                self.Pruning,
                lr = 0.003,
                weight_decay = self.Pruning.params.weight_decay,
                centripetal_force = 0.003,
            )
            if self.epoch == 110:
                break

        # TOT HIER

        # Prune each all filters in each cluster except for the first indexed filter
        layer = 0
        allowedlayer = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                if (self.Pruning.dependencies[layer][2] == True):
                    clustercount = -1
                    for cluster in clusterlist[allowedlayer]:
                        clustercount += 1
                        if len(cluster) < 2:
                            continue
                        combineFilters(layer, cluster, self.Pruning)

                        # Adjust integers in other clusters to account for shifts in tensor because of deleted filters
                        otherclustercount = 0
                        for othercluster in clusterlist[allowedlayer]:
                            if otherclustercount > clustercount:
                                for i in cluster:
                                    itemcount = 0
                                    for item in othercluster:
                                        if item >= i:
                                            othercluster[itemcount] -= 1
                                        itemcount +=1 
                            otherclustercount += 1
                        
                    allowedlayer += 1 
                layer += 1

        # check final amount of filters
        finalcount = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                filtersperlayerpruned.append(m.out_channels)
                finalcount += m.out_channels

        deleteGrads(self.Pruning)

        logstring = "The final amount of filters after pruning is " +  str(finalcount)
        self.logprune.info(logstring)
        self.prunedfilters = totalfilters - finalcount
        logstring = str(self.Pruning.manner) + " pruned " + str(self.prunedfilters) + " filters"
        self.logprune.info(logstring)
        self.logprune.info("Filters before pruning:")
        self.logprune.info(filtersperlayer)
        self.logprune.info("Filters after pruning:")
        self.logprune.info(filtersperlayerpruned)


    def clusterInLayer(self, layer):
        count = 0
        for m in self.Pruning.params.network.modules():
            if isConvolutionLayer(m):
                if count != layer:
                    count += 1
                    continue
                clusterAmount = int(((100.0-self.Pruning.percentage) * m.out_channels) / 100.0)
                return self.makeClusters(m, clusterAmount)


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
            #logstring = "Clusters: " + str(clusters)
            #self.logprune.info(logstring)
            return clusters
        elif (self.clustermethod == 'kmeans'):
            centroids = []
            for i in range(clusterAmount):
                centroids.append(m.weight.data[i].clone().detach())
            clusters = self.kmeansClustering(m, clusters, centroids)
            #logstring = "Clusters: " + str(clusters)
            #self.logprune.info(logstring)
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
