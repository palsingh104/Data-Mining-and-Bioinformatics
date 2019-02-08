from random import *
import time
from math import *
from collections import defaultdict

def loadfromfile(f, delim='\t'):
    fr = open(f)
    stringvector = [line.strip().split(delim) for line in fr.readlines()]
    return stringvector
def convert(sequence):
    for item in sequence:
        try:
            yield float(item)
        except ValueError as e:
            yield item

def validation(dataset):
    splits = list()
    datasetCopy = list(dataset)
    chunkSize = len(datasetCopy) // kFolds
    leftOver = len(datasetCopy) % kFolds
    start = 0
    for i in range(kFolds):
        if i < leftOver:
            end = start + chunkSize + 1
        else:
            end = start + chunkSize
        splits.append(datasetCopy[start:end])
        start = end
    return splits

def Matfunc(X0, Y1):
    TrueP,TrueN,FalseP,FalseN = 0,0,0,0
    for i in range(len(X0)):
        if X0[i] == 1 and Y1[i] == 1:
            TrueP = TrueP + 1
        if X0[i] == 1 and Y1[i] == 0:
            FalseN = FalseN + 1
        if X0[i] == 0 and Y1[i] == 1:
            FalseP = FalseP + 1
        if X0[i] == 0 and Y1[i] == 0:
            TrueN = TrueN + 1

    P,R,F1,ACC = 0,0,0,0
    if TrueP + FalseP != 0:
        P = TrueP / float(TrueP + FalseP)
    if TrueP + FalseN != 0:
        R = TrueP / float(FalseN + TrueP)
    if TrueP + TrueN + FalseP + FalseN != 0:
        ACC = (TrueP + TrueN) / float(TrueP + TrueN + FalseP + FalseN)
    if P + R != 0:
        F1 = (2 * P * R) / float(P + R)
    P *= 100
    R *= 100
    ACC *=100
    F1 *=100
    return {'P': P, 'R': R, 'ACC': ACC, 'F1': F1}

def splita(a, b, c):
    l ,r = [],[]
    _ = [l.append(row) if row[a] < b else r.append(row) for row in c]
    return l, r

def classinfo(a):
    return [row[-1] for row in a]

def calculateGINI(left, right, classes):
    n_instances = float(len(left) + len(right))
    gini = 0.0

    score = 0.0
    size = float(len(left))
    if size != 0:
        for class_val in classes:
            tempList = classinfo(left)
            p = tempList.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    score = 0.0
    size = float(len(right))
    if size != 0:
        for class_val in classes:
            tempList = classinfo(right)
            p = tempList.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    return gini

def getSplit(dataset, isRandomForest):
    classes = list(set(classinfo(dataset)))
    indexesList = []

    if isRandomForest:
        indexesList = selectFeatures(len(dataset[0]) - 1)
    else:
        indexesList = [x for x in range(len(dataset[0]) - 1)]

    index, value, score, left, right = inf, inf, inf, None, None

    for i in indexesList:
        for row in dataset:
            l, r = splita(i, row[i], dataset)
            gini = calculateGINI(l, r, classes)
            if gini <= score:
                index, value, score, left, right = i, row[i], gini, l, r
    return {'index': index, 'value': value, 'leftHalf': left, 'rightHalf': right}


def terminal(group):
    outcomes = classinfo(group)
    d = defaultdict(float)
    for i in outcomes:
        d[i] += 1
    result = max(d.items(), key=lambda x: x[1])
    return result[0]


def nodeSplit(node, depth, isRandomForest):
    left, right = node['leftHalf'], node['rightHalf']

    del (node['leftHalf'])
    del (node['rightHalf'])

    if not left or not right:
        node['left'] = node['right'] = terminal(left + right)
        return

    if depth >= maxDepth:
        node['left'], node['right'] = terminal(left), terminal(right)
        return

    if len(left) <= minSize:
        node['left'] = terminal(left)
    else:
        node['left'] = getSplit(left, isRandomForest)
        nodeSplit(node['left'], depth + 1, isRandomForest)

    if len(right) <= minSize:
        node['right'] = terminal(right)
    else:
        node['right'] = getSplit(right, isRandomForest)
        nodeSplit(node['right'], depth + 1, isRandomForest)

def buildTree(train, isRandomForest):
    root = getSplit(train, isRandomForest)
    nodeSplit(root, 1, isRandomForest)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def makeDecisionTree(train, test):
    tree = buildTree(train, False)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return (predictions), tree


def selectFeatures(currDataLength):
    indexesList = [x for x in range(currDataLength)]
    shuffle(indexesList)
    return indexesList[:featuresCount]


def makeSubSample(dataset):
    sample = list()
    size = len(dataset)
    sampleCount = round(size * samplingRatio)
    indexes = [randint(0, size - 1) for x in range(sampleCount)]
    for i in indexes:
        sample.append(dataset[i])
    return sample

def baggingPrediction(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def makeRandomForest(train, test, treesCount):
    trees = list()
    for i in range(treesCount):
        sample = makeSubSample(train)
        tree = buildTree(sample, True)
        trees.append(tree)
    predictions = [baggingPrediction(trees, row) for row in test]
    return (predictions)

seed(11)
f = 'project3_dataset2.txt'# specify file
datasetAll = loadfromfile(f)
data = [list(convert(sublist)) for sublist in datasetAll]
kFolds = 10
maxDepth = 3
minSize = 1
featuresCount = int(sqrt(len(data[0]) - 1))
samplingRatio = 1.0

#start_time = time.time()
splits = validation(data)
scores = []
for split in splits:
    trainSet = list(splits)
    trainSet.remove(split)
    trainSet = sum(trainSet, [])
    testSet = []
    actual = []
    for row in split:
        rowCopy = list(row)
        testSet.append(rowCopy)
        rowCopy[-1] = None
        actual.append(row[-1])
    predicted,tree = makeDecisionTree(trainSet, testSet)
        #print(tree)
    accuracy = Matfunc(actual, predicted)
        #print(accuracy)
    scores.append(accuracy)

P,R,ACC,F1 = 0,0,0,0

for i in range(len(scores)):
    P += scores[i]['P']
    R += scores[i]['R']
    ACC += scores[i]['ACC']
    F1 += scores[i]['F1']

print('MeanP =>: %.2f%%' % (P / float(len(scores))))
print('MeanR => %.2f%%' % (R / float(len(scores))))
print('MeanACC =>: %.2f%%' % (ACC / float(len(scores))))
print('MeanF1 => %.3f%%' % (F1 / float(len(scores))))

# Ref
#https://towardsdatascience.com/
#https://machinelearningmastery.com/
##stackoverflow and few other sources for help