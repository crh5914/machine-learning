import numpy as np
def createDataSet():
    data = np.array([[2,2],[3,2],[2,3],[1,0],[1,1],[0,1]])
    labels = [1,1,1,0,0,0]
    return data,labels
def classify(inX,dataSet,labels,k):
    diff = dataSet-inX
    diff = diff**2
    diff = np.sum(diff**0.5,axis=1)
    indexs = np.argsort(diff,axis=0)
    labelCount = {}
    for i in range(k):
        label = labels[indexs[i]]
        labelCount[label] = labelCount.get(label,0) + 1
    labelCount = sorted(labelCount.items(),key = lambda x:x[1],reverse=True)
    return labelCount[0][0]
def run():
    dataSet,labels = createDataSet() 
    x = np.array([0,0])
    y = classify(x,dataSet,labels,2)
    print(y)
    x = np.array([3,3])
    y = classify(x,dataSet,labels,2)
    print(y)
run()