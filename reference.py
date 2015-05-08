# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

reference: http://nbviewer.ipython.org/github/justmarkham/gadsdc1/blob/master/logistic_assignment/kevin_logistic_sklearn.ipynb
"""
from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

datasize = 1500

initcolumns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x']
listData = range(0,21)
#print randomSelect

dummies = pd.get_dummies(listData)
#print dummies[randomSelect]

df = pd.DataFrame(columns=initcolumns, index = np.arange(datasize))

#print df

for num in range(0,datasize):
    randomSelect = np.random.choice(listData)
    randomData = np.array(dummies[randomSelect], int)
    #print randomData
    #print dummies[randomSelect]
    groundtruthValue = np.random.randint(0, 2, 1)
    r = np.random.randint(1, 6, 1)
    f = np.random.randint(1, 6, 1)
    rdata = np.insert(randomData, 21, r, axis=0)
    rdata = np.insert(rdata, 22, f, axis=0)
    rdata = np.insert(rdata, 23, groundtruthValue, axis=0)
    #print rdata
    df.ix[num] = rdata
    #print type(dummies[randomSelect])
    
#print df    

y, X = dmatrices('x ~ a + b + c + d + e + f + g + h+ i+ j+ k+ l+ m+ n+ o+ p+ q+ r+ s+ t+ u+ v+ w',
                  df, return_type="dataframe")

y = np.ravel(y)

model = LogisticRegression()
model = model.fit(X, y)
print model.score(X, y); #0.89453
print y.mean(); #0.10546

#w big test
print 'test~~~'
#df20 = pd.read_csv('out_gt/test_data', names = ['a', 'b', 'c', 'd', 'e'])
df20 = pd.read_csv('out_gt/day20', names = initcolumns)
#print df20
#y20, X20 = dmatrices('e ~ a + b + c + d', df20, return_type='dataframe')
y20, X20 = dmatrices('x ~ a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w',
                  df20, return_type = "dataframe")
#print X20
#print y20
y20 = np.ravel(y20)
#print y20
model20 = LogisticRegression()
model20 = model20.fit(X20, y20)
#print y20.mean()
print model20.score(X20, y20) #accuracy rate = (TP+TN)/sample_size (note:probability=0.5)
#print model20.predict_proba(X20) #type: numpy.ndarray
#print model20.predict_proba([1,0,0,0,0]) #[Intercept=1, a, b, c, d]
df20_proba = model20.predict_proba(X20)
#print df20_proba[:,1]
print df20_proba[:,1][0]
print df20_proba[:,1][1]
print df20_proba[:,1][2]
TP = 0
FP = 0
TN = 0
FN = 0
threshold = 0.00
data_size = 0
while threshold < 0.01: # from 0.00 to 0.40
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    data_size = 0
    #with open('out_gt/test_data') as fp:
    with open('out_gt/day20') as fp:
        for line in fp:
            gt = line.split(",")[-1][0] #last index of line
            proba = df20_proba[:,1][data_size]
            #pred = 'true' if float(proba) > threshold else 'false'
            if float(proba) >= threshold and int(gt) == 1:
                TP += 1
            if float(proba) >= threshold and int(gt) == 0:
                FP += 1
            if float(proba) < threshold and int(gt) == 1:
                FN += 1
            if float(proba) < threshold and int(gt) == 0:
                TN += 1
            data_size += 1
            #print 'gt:%s, proba:%s, pred:%s' % (gt, proba, pred)
    #print 'TP %d, FP %d, FN %d, TN %d' % (TP, FP, FN, TN)
    print 'threshold %f, TPR %.2f, FPR %.2f, accuracy %f' % (threshold, TP/(TP + FN), FP/(FP + TN), (TP + TN)/data_size)
    threshold += 0.01
print df20_proba[:,0].size
print data_size
print TP+FP+TN+FN
print 'test~~~'
#w big test

#w big test 2
print 'test2~~~'
df21 = pd.read_csv('out_gt/day21', names = initcolumns)
y21, X21 = dmatrices('x ~ a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w',
                     df21, return_type = "dataframe")
y21 = np.ravel(y21)
print y21.size
df21_proba = model20.predict_proba(X21)
threshold21 = 0.00
while threshold21 < 0.40:
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    data_size = 0
    with open('out_gt/day21') as fp:
        for line in fp:
            gt = line.split(",")[-1][0]
            proba = df21_proba[:,1][data_size]
            if float(proba) >= threshold21 and int(gt) == 1:
                TP += 1
            if float(proba) >= threshold21 and int(gt) == 0:
                FP += 1
            if float(proba) < threshold21 and int(gt) == 1:
                FN += 1
            if float(proba) < threshold21 and int(gt) == 0:
                TN += 1
            data_size += 1
    print 'threshold %f, TPR %.2f, FPR %.2f, accuracy %f, data_size %d' % (threshold21, TP/(TP + FN), FP/(FP + TN), (TP + TN)/data_size, data_size)
    threshold21 += 0.01
print 'test2~~~'
#w big test 2

#print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))


#predict a given input data
prevalue = model.predict_proba(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]))
print 'acc:%s' % prevalue


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
#print model2


# predict class labels for the test set
predicted = model2.predict(X_test)
#print predicted


# generate class probabilities
probs = model2.predict_proba(X_test)
#print probs



# generate evaluation metrics
print 'acc:%f' % metrics.accuracy_score(y_test, predicted)
print 'roc_auc:%f' % (metrics.roc_auc_score(y_test, probs[:, 1]))

print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)



# evaluate the model using 10-fold cross-validation
#scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
#print scores
#print scores.mean()

result = model.predict_proba(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0]))
print result

