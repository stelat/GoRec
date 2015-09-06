import numpy as np
import cv2
from sklearn import svm, cross_validation
import pickle
from sklearn.externals import joblib
from extract_feature import *
from os import listdir
from os.path import isfile, join
from random import shuffle

def data_shuffle(X,Y):
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(X))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(X[i])
        list2_shuf.append(Y[i])
    return list1_shuf, list2_shuf

def read_data(path):
    X=[]
    Y=[]
    empty_path = path + "/empty"
    white_path = path + "/white"
    black_path = path + "/black"
    paths = [empty_path, white_path, black_path]

    for i,path in enumerate(paths):
        onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
        for image in onlyfiles:
            image_array = cv2.imread(path + "/" + image)
            X.append(extract_feature(image_array))
            Y.append(i)

    X,Y = data_shuffle(X,Y)

    return (np.asarray(X),np.asarray(Y))

#http://scikit-learn.org/stable/modules/svm.html
def trainSVM(X,y):
    clf = svm.SVC(kernel='linear')
    #try different kernel
    clf.fit(X, y)
    return clf
   
def testSVM(testDataSetpath,clf):
    for image in testDataSetpath:
        if label(image)==clf.predict(extract_feature(image)):
            ok+=1
        else:
            ko+=1
    print "accuracy:" + float( ok)/(ok+ko)






#main:
(X,Y)=read_data("data")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
print X.shape
print Y
print scores
#clf=trainSVM(X,Y)

#save the SVM parameters to use it later
#joblib.dump(clf, 'linearSVM.pkl')
#testSVM("dataTest",clf)
