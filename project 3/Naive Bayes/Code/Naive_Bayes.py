import pandas as pd
import numpy as np

data = pd.read_csv('pppp.txt', sep="\t")
print(data.head)
num_of_cols = len(data.columns)
#print('num_of_cols: ', num_of_cols)
#print('num_of_rows: ', len(data))
col_names = []
for i in range(num_of_cols):
    col_names.append('feature' + str(i))

col_names[num_of_cols - 1] = 'result'
data.columns = col_names

# print(data.head)

def count(data, colname, label, target):
    condition = (data[colname] == label) & (data['result'] == target)
    return len(data[condition])


predicted = []
probabilities = {0: {}, 1: {}}

train_percent = 100
train_len = int((train_percent * len(data)) / 100)
train_X = data.iloc[:train_len, :]
# test_X = data.iloc[train_len+1:,:-1]
# test_y = data.iloc[train_len+1:,-1]

demo_X = [['sunny', 'cool', 'high', 'weak']]
demo_test_X = pd.DataFrame(demo_X)
# print(demo_test_X)
demo_test_X.columns = col_names[0: num_of_cols - 1]
print('Test data')
print(demo_test_X)

# print(train_X.head)
count_0 = count(train_X, 'result', 0, 0)
count_1 = count(train_X, 'result', 1, 1)
# print(count_0)
# print(count_1)

prob_0 = count_0 / len(train_X)
prob_1 = count_1 / len(train_X)
# print(prob_0)
# print(prob_1)

for col in train_X.columns[:-1]:
    probabilities[0][col] = {}
    probabilities[1][col] = {}
    labels = np.unique(data[col])

    for category in labels:
        count_ct_0 = count(train_X, col, category, 0)
        count_ct_1 = count(train_X, col, category, 1)

        probabilities[0][col][category] = count_ct_0 / count_0
        probabilities[1][col][category] = count_ct_1 / count_1

# print(probabilities)


for row in range(0, len(demo_test_X)):
    pred_0 = prob_0
    print(prob_0)
    pred_1 = prob_1
    print(prob_1)
    for feature in demo_test_X.columns:
        pred_0 *= probabilities[0][feature][demo_test_X[feature].iloc[row]]
        pred_1 *= probabilities[1][feature][demo_test_X[feature].iloc[row]]

    # Predict the outcome
    if pred_0 > pred_1:
        predicted.append(0)
    else:
        predicted.append(1)

print('Prediction on test data', predicted[0])
print('P(H1 / X) + P(H0 / X): ', predicted[0])

# tp,tn,fp,fn = 0,0,0,0
# for j in range(0,len(predicted)):
#     if predicted[j] == 0:
#         if test_y.iloc[j] == 0:
#             tp += 1
#         else:
#             fp += 1
#     else:
#         if test_y.iloc[j] == 1:
#             tn += 1
#         else:
#             fn += 1

# print('Accuracy: ', ((tp+tn)/1)*100)

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