import numpy as np
import math
import pandas as pd
import operator


def loadDataSet(file_name, split_ratio=0.8):
    data = pd.read_csv(file_name, sep="\t")
    data = data.replace('Present|Absent', 0, regex=True)
    data = np.array(data.values)

    num_observations = len(data)
    num_of_cols = len(data[0])
    num_of_features = num_of_cols - 1
    # print('num_observations: ', num_observations)
    # print('num_of_cols: ', num_of_cols)

    num_of_training = int(split_ratio * num_observations)

    # print('num_of_training', num_of_training)
    num_of_test = num_observations - num_of_training
    # print('num_of_test', num_of_test)

    np.random.shuffle(data)
    print(data.shape)
    train_set = data[0: num_of_training, :]
    #     train_set  = np.array(data[0 : num_of_training, 0: num_of_features])
    # print('Num of training data: ', train_set.shape)
    train_labels = train_set[:, num_of_features:]
    # print('Train data labels: ', train_labels.shape)

    test_set = np.array(data[num_of_training:num_observations, :])
    #     test_set = np.array(data[num_of_training:num_observations, 0: num_of_features])
    # print('Num of test data', test_set.shape)
    test_labels = test_set[:, num_of_features:]
    # print('Test data labels: ', test_labels.shape)

    return train_set, train_labels, test_set, test_labels


def computeEuclideanDist(data1, data2):
    distance = 0.0
    for idx in range(len(data1)):
        distance += pow((data1[idx] - data2[idx]), 2)

    #     print('distance', distance)
    return math.sqrt(distance)

'''def Matfunc(X0, Y1):
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
    return {'P': P, 'R': R, 'ACC': ACC, 'F1': F1}'''

def computeKNN(train_data, test_data, k):
    #     test_data = np.array(test_data)
    num_of_cols = len(train_data[0])
    result = []
    for i in test_data:
        distances = []
        knn = []
        good = 0
        bad = 0
        for j in train_data:
            dist = computeEuclideanDist(i, j)
            distances.append((j[num_of_cols - 1], dist))
            distances.sort(key=operator.itemgetter(1))
            knn = distances[: k]
            #             print(knn)
            for val in knn:
                if val[0] == 1:
                    good += 1
                else:
                    bad += 1

        #         print(i.shape)
        if good > bad:
            i = np.append(i, 1)
        elif good < bad:
            i = np.append(i, 0)
        else:
            i = np.append(i, np.nan)

        result.append(i)
    return result


def computeAccuracy(result):
    num_of_correct_predictions = 0.0
    num_cols = len(result[0])
    #     print(num_cols)
    #     print(result[0][: 3])
    for res in result:
        if res[num_cols - 1] == res[num_cols - 2]:
            num_of_correct_predictions += 1

    accuracy = (float(num_of_correct_predictions) / len(result)) * 100
    print('k-NN accuracy: ', accuracy)


################## DEMO TESTING CODE BELOW ########################

k = 10
train_set, train_labels, dummy, dummy = loadDataSet('project3_dataset3_train.txt', 1)
test_set, test_labels, dummy, dummy = loadDataSet('project3_dataset3_test.txt', 1)
# print(test_labels)

num_of_features = len(train_set[0]) - 1

result = computeKNN(train_set, test_set, k)
computeAccuracy(result)

### this is just to cross check the performance of custom implementation of K-NN
from sklearn.neighbors import KNeighborsClassifier

X = np.array(train_set[:, 0: num_of_features])
y = np.ravel(train_labels)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

test_X = np.array(test_set[:, 0: num_of_features])
sklearn = neigh.predict(test_X)

num_of_correct_predictions = 0.0
for i in range(len(test_labels)):
    if sklearn[i] == test_labels[i]:
        num_of_correct_predictions += 1

accuracy = (float(num_of_correct_predictions) / len(test_labels)) * 100
print('k-NN accuracy: ', accuracy)

'''splits = validation(data)
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
print('MeanF1 => %.3f%%' % (F1 / float(len(scores))))'''


#modification and commenting are subject to makiung code run on specific and desired data sets