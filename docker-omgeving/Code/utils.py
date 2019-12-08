#
#   Utility functions
#   

# Basic imports
import lightnet as ln
import torch
import numpy as np

# Settings
ln.logger.setConsoleLevel('ERROR')  # Only show error log messages


# Iterate through a list and prune the given filters
# ATTENTION: the order of given filters matters
def hardPruneFilters(Pruning, prunelist):
    for filter in prunelist:
        #print('Hard pruning filter', filter[1], '@ layer', filter[0])
        layer = 0
        done = False
        combinedLayer = 0
        for m in Pruning.params.network.modules():
            if isConv2dBatchRelu(m):
                if (combinedLayer == Pruning.dependencies[filter[0]][3]):
                    m.out_channels -= 1
                combinedLayer += 1
            if done == True:
                if isBatchNormalizationLayer(m):
                    m.num_features -= 1
                    if m.running_mean is not None:
                        m.running_mean = torch.cat((m.running_mean[:filter[1]], m.running_mean[filter[1]+1:]))
                    if m.running_var is not None:
                        m.running_var = torch.cat((m.running_var[:filter[1]], m.running_var[filter[1]+1:]))
                    if m.weight is not None:
                        m.weight.data = torch.cat((m.weight.data[:filter[1]], m.weight.data[filter[1]+1:]))
                    if m.bias.data is not None:
                        m.bias.data = torch.cat((m.bias.data[:filter[1]], m.bias.data[filter[1]+1:]))
                    break
                else:
                    break
            if isConvolutionLayer(m):
                if layer != filter[0]:
                    layer += 1
                    continue

                if m.bias is not None:
                    m.bias.data = torch.cat((m.bias.data[:filter[1]], m.bias.data[filter[1]+1:]))

                m.weight.data = torch.cat((m.weight.data[:filter[1]], m.weight.data[filter[1]+1:]))
                m.out_channels -= 1

                if len(Pruning.dependencies[layer][1]) != 0:
                    for dependency in Pruning.dependencies[layer][1]:
                        prunetuple = (dependency, filter[1])
                        pruneFeatureMaps(Pruning, [prunetuple])
                done = True
    return


# Iterate through a list and prune the given feature maps
# ATTENTION: the order of given feature maps matters
def pruneFeatureMaps(Pruning, prunelist):
    for featuremap in prunelist:
        #print('Hard pruning feature map', featuremap[1], '@ layer', featuremap[0])
        layer = 0
        combinedLayer = 0
        for m in Pruning.params.network.modules():
            if isConv2dBatchRelu(m):
                if (combinedLayer == Pruning.dependencies[featuremap[0]][3]):
                    m.in_channels -= 1
                combinedLayer += 1
            if isConvolutionLayer(m):
                if layer != featuremap[0]:
                    layer += 1
                    continue
                m.weight.data = torch.cat((m.weight.data[:,:featuremap[1]], m.weight.data[:,featuremap[1]+1:]), 1)
                m.in_channels -= 1
                break
        return


# Iterate through a list and prune the given filters
# ATTENTION: the order of given filters matters
def softPruneFilters(Pruning, model, prunelist):
    for filter in prunelist:
        #print('Soft pruning filter', filter[1], '@ layer', filter[0])
        done = False
        layer = 0
        for m in model.modules():
            if done == True:
                if isBatchNormalizationLayer(m):
                    m.num_features -= 1
                    if m.running_mean is not None:
                        m.running_mean[filter[1]] = 0
                    if m.running_var is not None:
                        m.running_var[filter[1]] = 0
                    if m.weight is not None:
                        m.weight.data[filter[1]] = 0
                    if m.bias.data is not None:
                        m.bias.data[filter[1]] = 0
                    break
                else:
                    break
            if isConvolutionLayer(m):
                if layer != filter[0]:
                    layer += 1
                    continue

                if m.bias is not None:
                    m.bias.data[filter[1]] = 0

                zeros = torch.zeros([1,m.weight.data.shape[1],m.weight.data.shape[2], m.weight.data.shape[3]], device=Pruning.device)
                buffer = torch.cat((m.weight.data[:filter[1]], zeros))
                m.weight.data = torch.cat((buffer, m.weight.data[filter[1]+1:]))
                done = True
        return


def deleteGrads(Pruning):
    for m in Pruning.params.network.modules():
        if isBatchNormalizationLayer(m):
            if m.running_mean is not None:
                    m.running_mean.grad = None
            if m.running_var is not None:
                    m.running_var.grad = None
            if m.weight is not None:
                    m.weight.grad = None
            if m.bias.data is not None:
                    m.bias.grad = None
        elif isConvolutionLayer(m):
            if (m.bias is not None):
                m.bias.grad = None
            m.weight.grad = None


# Find lowest non-zero value in a non-negative array
def arg_nonzero_min(a):
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


def isConvolutionLayer(module):
    if isinstance(module, torch.nn.Conv2d):
        return True
    else:
        return False

    
def isBatchNormalizationLayer(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        return True
    else:
        return False


def isConv2dBatchRelu(module):
    if isinstance(module, ln.network.layer.Conv2dBatchReLU):
        return True
    else:
        return False


# Make a list of dependencies of convolutional layers
def makeDependencyList(model):
    randomInput = torch.rand(1, 3, 416, 416)
    traced_cell = torch.jit.trace(model, randomInput)
    traced_cell_output = traced_cell.code

    listed_trace = [s.strip() for s in traced_cell_output.splitlines()] # Split trace in lines

    convolutionlist = []
    index = 0
    for text in listed_trace:
        if "torch._convolution" in text: # If line contains a convolution link it to its index
            layername = text.split(" = ")[0]
            convolutionlist.append((index, layername)) 
            index+=1

    dependencylist = []
    for conv in convolutionlist:
        dependants = findDependants(conv[1], listed_trace, convolutionlist)
        dependencylist.append([conv, transformDependants(dependants, convolutionlist)])     

    allowedToPrune(dependencylist)

    findConv2dBatchReluLayers(dependencylist, model)

    return dependencylist


def findConv2dBatchReluLayers(dependencylist, model):
    isCombinedLayer = False
    combinedcount = -1
    convcount = 0
    for m in model.modules():
        if isConv2dBatchRelu(m):
            combinedcount += 1
            isCombinedLayer = True
        if isConvolutionLayer(m):
            if isCombinedLayer == True:
                dependencylist[convcount].append(combinedcount)
                isCombinedLayer = False
            else:
                dependencylist[convcount].append(None)
            convcount += 1


# If the dependant is also dependant of another layer you cannot prune in this layer
def allowedToPrune(dependencylist):
    count = -1
    for element in dependencylist:
        count += 1
        if count == len(dependencylist)-1:
            element.append(False)
            break
        element.append(True)
        for other in dependencylist:
            if element[0][0] != other[0][0]:
                for dependant in element[1]:
                    if dependant in other[1] :
                        element[2] = False
                        break


# Transforms the layer indexes so it is in the same order as in the Pruning class
def transformDependants(dependants, convolutionlist):
    newList = []
    done = []
    for dependant in dependants:
        if dependant in done:
            continue
        for conv in convolutionlist:
            if (dependant == conv[1]):
                newList.append(conv[0])
                done.append(dependant)
    return newList


# Split text on multiple different signs
def splitTextOnSign(text):
    words = []
    word = ""
    for x in text:
        if x == "(" or x == ")" or x == "." or x == "," or x == " " or x == "]" or x == "[":
            words.append(word)
            word=""
        else:
            word += x
    words.append(word)
    return words


# See if a given text contains a word
def textContainsWord(text, word):
    words = splitTextOnSign(text)
    for w in words:
        if w == word:
            return True
    return False


 # Find the next convolutionlayer that is dependant of the given layer
def findConvDependant(layername, listed_trace, convolutionlist, dependants):
    for conv in convolutionlist:
        if conv[1] == layername:
            return conv[1]
    for text in listed_trace:
        if "=" in text:
            substring = text.split(" = ")
            if textContainsWord(substring[1], layername):
                dependants.append(findConvDependant(substring[0], listed_trace, convolutionlist, dependants))
    return


# Find the dependants of a given convolutionlayer
def findDependants(layername, listed_trace, convolutionlist): 
    dependants = []
    for text in listed_trace:
        if "=" in text:
            substring = text.split(" = ")
            if textContainsWord(substring[1], layername):
                findConvDependant(substring[0], listed_trace, convolutionlist, dependants)
    return dependants
