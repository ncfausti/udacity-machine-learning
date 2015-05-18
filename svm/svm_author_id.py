#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import time
sys.path.append("../tools/")
sys.path.append("../choose_your_own/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
from sklearn.svm import SVC
from class_vis import prettyPicture
c = 10000
clf = SVC(kernel="rbf", C=c)

print("RBF and c=%d"%c)
#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

#slice training sets to 1% of original size
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t = time.time()
clf.fit(features_train,labels_train)
t2 = time.time() - t
print("Training took %.2f seconds" % t2)

#### store your predictions in a list named pred
t = time.time()
pred = clf.predict(features_test)
t2 = time.time() - t
print("Prediction took %.2f seconds" % t2)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuracy: %.6f" % acc)
print("10:%d, 26:%d, 50:%d" % (pred[10],pred[26],pred[50]))
#prettyPicture(clf, features_test, labels_test)

chris_count = 0
for i in range(len(pred)):
	if pred[i] == 1:
		chris_count += 1

print("number of emails from chris: %d" % chris_count)
#########################################################
"""


# THIS IS THE TERRAIN DATA for SVM, *NOT* email data for SVM mini-project

import sys
import time
sys.path.append("../tools/")
sys.path.append("../choose_your_own/")
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=900000.0)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
t = time.time()
clf.fit(features_train,labels_train)
t2 = time.time() - t
print("Training took %.2f seconds" % t2)

#### store your predictions in a list named pred
t = time.time()
pred = clf.predict(features_test)
t2 = time.time() - t
print("Prediction took %.2f seconds" % t2)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print("Accuracy: %.6f" % acc)
prettyPicture(clf, features_test, labels_test)


def submitAccuracy():
    return acc
"""