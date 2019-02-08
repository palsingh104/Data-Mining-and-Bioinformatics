from random import *
from math import *
from collections import defaultdict
import numpy
import random

input = 'project3_dataset2.txt'

def main(input):
    crossvalfold = 10
    maxDepth, minSize = 1, 1
    inputdataset = loadfromfile(input)
    data = [list(convert(sublist)) for sublist in inputdataset]
    kf = 10 #folds req
    start = 0
    sets = len(data) // kf
    rest = len(data) % kf
    ACC, P, R, fnm = [], [], [], []

    for i in range(0, crossvalfold):

        testdata = data
        traindata = data

        end = start + sets

        if i == crossvalfold - 1:
            end = end + rest

        testdata = testdata[start:end]

        trainingdata = numpy.delete(traindata, numpy.s_[start:end], axis=0)

        predicted_label, test_labels = adaboost(testdata, traindata, 5, maxDepth, minSize)
        EVALUATEDresult = Matfunc(test_labels, predicted_label)
        R.append(EVALUATEDresult['R'])# print(EVALUATEDresult)
        P.append(EVALUATEDresult['P'])
        fnm.append(EVALUATEDresult['F1'])
        ACC.append(EVALUATEDresult['ACC'])

        start = end

    print("MeanACC =>", numpy.mean(ACC))  # using numpy numpy
    print("MeanP =>", numpy.mean(P))
    print("MeanR => ", numpy.mean(R))
    print("MeanF1 => ", numpy.mean(fnm))

def loadfromfile(input, delim='\t'):
    fr = open(input)
    result = [line.strip().split(delim) for line in fr.readlines()]
    return result

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

def convert(sequence):
    for item in sequence:
        try:
            yield float(item)
        except ValueError as e:
            yield item
def adaboost(testdata,traindata,iter,maxDepth,minSize):

    def splitOnValue(index, value, dataset):
        left = []
        right = []
        _ = [left.append(row) if row[index] < value else right.append(row) for row in dataset]
        return left, right

    def getClasses(dataset):
        return [row[-1] for row in dataset]

    def calculateGINI(left, right, classes):
        n_instances = float(len(left) + len(right))
        gini = 0.0

        score = 0.0
        size = float(len(left))
        if size != 0:
            for class_val in classes:
                tempList = getClasses(left)
                p = tempList.count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)

        score = 0.0
        size = float(len(right))
        if size != 0:
            for class_val in classes:
                tempList = getClasses(right)
                p = tempList.count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)

        return gini

    def getSplit(dataset, isRandomForest):
        classes = list(set(getClasses(dataset)))
        indexesList = []

        indexesList = [x for x in range(len(dataset[0]) - 1)]

        index, value, score, left, right = inf, inf, inf, None, None

        for i in indexesList:
            for row in dataset:
                l, r = splitOnValue(i, row[i], dataset)
                gini = calculateGINI(l, r, classes)
                if gini < score:
                    index, value, score, left, right = i, row[i], gini, l, r
        return {'index': index, 'value': value, 'leftHalf': left, 'rightHalf': right}


    def terminal(group):
        outcomes = getClasses(group)
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
        return predictions,tree


    def geTrueNone(testdata):
        for i in range(len(testdata)):
            data[i][-1] = None
        return data

    def getLabels(data):
        labels = []
        for i in range(len(data)):
            labels.append(data[i][-1])
        return labels

    def getmisses(predicted,trainlabels):
        miss = []
        miss2 = []
        for i in range(len(predicted)):
            if predicted[i] != trainlabels[i]:
                miss.append(1)
                miss2.append(1)
            else:
                miss.append(0)
                miss2.append(-1)
        return miss,miss2

    def getindices(data):
        indices_list = []
        for i in range(len(data)):
            indices_list.append(i)
        return indices_list

    def getrandindex(indices,w):
        rand_indices = []
        for i in range(len(indices)):
            rand_ind = random.choices(indices,w)
            rand_indices.append(rand_ind[0])
        return rand_indices

    def getrandRows(data,rand_indices):
        rows = []
        for ind in rand_indices:
            rows.append(data[ind])
        return rows


    w = numpy.ones(len(traindata))/len(traindata)

    trainlabels = getLabels(traindata)
    train_indices = getindices(traindata)

    trees = []
    alpha = []

    j = 0

    while j < iter:
        if j > 0 :
            rand_indices = getrandindex(train_indices,w)
            traindata = getrandRows(traindata,rand_indices)

        predicted,tree = makeDecisionTree(traindata, traindata)

        miss,miss2 = getmisses(predicted,trainlabels)

        err_m = numpy.dot(w,miss)/sum(w)

        alpha_m = 0.5*numpy.log(float(1-err_m)/float(err_m))
        alpha.append(alpha_m)

        w = numpy.multiply(w,numpy.exp([float(x) * alpha_m for x in miss2]))
        trees.append(tree)

        j = j + 1

    predicted_label = []
    for row in testdata:
        zero_alpha = []
        one_alpha = []
        for i in range(len(trees)):
            prediction = predict(trees[i], row)
            if prediction == 0.0:
                zero_alpha.append(alpha[i])
            else:
                one_alpha.append(alpha[i])
        if sum(zero_alpha) > sum(one_alpha):
            predicted_label.append(0.0)
        else:predicted_label.append(1.0)

    # print(predicted_label)
    test_label = getLabels(testdata)

    return predicted_label,test_label

main(input)

#References
#https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
#stackoverflow and few other sources for help