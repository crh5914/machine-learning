import numpy as np
import math
def createDataSet():
	data = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
	labels = ['no surfacing','flippers']
	return data,labels
def calShannonEntr(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for vect in dataSet:
		label = vect[-1]
		labelCounts[label] = labelCounts.get(label,0) + 1
	entr = 0
	for key in labelCounts:
		prob = labelCounts[label]/numEntries
		entr -= prob*math.log(prob,2)
	return entr
def splitDataSet(dataSet,axis,value):
	newSet = []
	for vect in dataSet:
		if vect[axis] == value:
			newVect = vect[:axis]
			newVect.extend(vect[axis+1:])
			newSet.append(newVect)
	return newSet
def chooseBestFeacture(dataSet):
	feactureNum = len(dataSet[0])-1
	samples = len(dataSet)
	feactures = {}
	initEntr = calShannonEntr(dataSet)
	maxEntr = 0
	bestFeacture = -1
	for k in range(feactureNum):
		values = [dataSet[i][k] for i in range(samples)]
		feactures[k] = set(values)
	for k in feactures:
		entr = 0
		for value in feactures[k]:
			subSet = splitDataSet(dataSet,k,value)
			entr += len(subSet)/samples*calShannonEntr(subSet)
		if initEntr-entr > maxEntr:
			bestFeacture = k
			maxEntr = initEntr-entr
	return bestFeacture
def createTree(dataSet,classlist):
	decisionTree = {}
    if len(classlist) == len(dataSet):
		
	best = chooseBestFeacture(dataSet)
	
def run():
	dataSet,labels = createDataSet()
	print(calShannonEntr(dataSet))
	print(splitDataSet(dataSet,1,1))
run()