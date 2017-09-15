import numpy as np
import math
def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['No Surfacing','Flippers']
    return dataSet,labels
def calShannonEntr(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    for vect in dataSet:
        label = vect[-1]
        labelCount[label] = labelCount.get(label,0) + 1
    entr = 0
    #print(labelCount)
    for key in labelCount:
        prob = labelCount[label]/numEntries
        entr -= prob*math.log(prob,2)
    return entr
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for vect in dataSet:
        #print(vect[axis] == value)
        if(vect[axis] == value):
            newVect = vect[:axis]
            newVect.extend(vect[axis+1:])
            #print(newVect)
            retDataSet.append(newVect)
    return retDataSet
def chooseBestFeactureToSplit(dataSet):
    feacture_num = len(dataSet[0])-1
    initEntropy = calShannonEntr(dataSet)
    infoGain = 0
    bestFeacture = -1
    for i in range(feacture_num):
        values = [example[i] for example in dataSet]
        uniqueValues = set(values)
        entropy = 0
        for uniqueValue in uniqueValues:
            subSet = splitDataSet(dataSet,i,uniqueValue)
            prob = len(subSet)/len(dataSet)
            entropy += prob*calShannonEntr(subSet)
        entropyGain = initEntropy-entropy
        if entropyGain > infoGain:
            bestFeacture = i
            infoGain = entropyGain
    return bestFeacture
def majorityCnt(classlist):
    classCount = {}
    for c in classlist:
        classCount[c] = classCount.get(c,0)+1
    classCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
    return classCount[0][0]

def createTree(dataSet,labels):
    classList = [ example[-1] for example in dataSet]
    #all samples in the same class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #no feature for further division 
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeactureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    featValues = set(featValues)
    for value in featValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree        
def classify(inputTree,labels,testVect):
    featNode = list(inputTree.keys())[0]
    feat = labels.index(featNode)
    secondLayer = inputTree[featNode]
    for value in secondLayer:
        if value == testVect[feat]:
            if isinstance(secondLayer[value],dict):
                return classify(secondLayer[value],labels,testVect)
            else:
                return secondLayer[value]
def run():
    dataSet,labels = createDataSet()
    tree = createTree(dataSet,labels[:])
    #check the tree
    print(tree)
    #test classify
    print(classify(tree,labels,[1,0]))
run()
    