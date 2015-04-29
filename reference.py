# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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
print model.score(X, y);
#print y.mean();

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

