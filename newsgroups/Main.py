__author__ = 'CJ'

import nltk
import random
from collections import defaultdict

newsgroups = nltk.corpus.PlaintextCorpusReader('./data/', '.*/[0-9]+', encoding='latin1')
ids = newsgroups.fileids()

random.seed(0)
random.shuffle(ids)
#ids = ids[:5000]  # Change for whole dataset
size = len(ids)

testSet = ids[:int(size*0.1)]
trainSet = ids[int(size*0.1):]

print("Length of training data: ", len(trainSet), "\nLength of test data: ", len(testSet))


def features(text):
    # Convert a post into a dictionary of features
    features = defaultdict(int)
    for word in text:
        if word.isalpha():
            features[word.lower()] += 1
    return features


def getclass(fileid):
    # Returns class from fileid
    return fileid.split('/')[0]

# Set training data and testData
trainData = [(features(newsgroups.words(fileids=f)),getclass(f)) for f in trainSet]
testData = [(features(newsgroups.words(fileids=f)),getclass(f)) for f in testSet]

c = nltk.FreqDist(item[1] for item in trainData)
#default = c.keys()[0]
default = (next(iter(c.values())))

print("c: ", c)

sum(c == default for f, c in testData) / float(len(testData))

# Naive Bayes
nb = nltk.NaiveBayesClassifier.train(trainData)
print("NB: ", nltk.classify.accuracy(nb, testData))


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
# BernoulliNB
bernoulli = SklearnClassifier(BernoulliNB())
bernoulli.train(trainData)
print("NB Bernoulli: ", nltk.classify.accuracy(bernoulli, testData))
gaussian = SklearnClassifier(GaussianNB())
gaussian.train(trainData.toarray(trainData))
print("Gaussian: ", nltk.classify.accuracy(gaussian, testData))

from sklearn.naive_bayes import MultinomialNB
# MultinomialNB
multi = SklearnClassifier(MultinomialNB())
multi.train(trainData)
print("NB Multinomial: ", nltk.classify.accuracy(multi, testData))

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])

pmulti = SklearnClassifier(pipeline)
pmulti.train(trainData)
print("NB Multinomial (pmulti): ", nltk.classify.accuracy(pmulti, testData))


from sklearn.metrics import f1_score

#results = pmulti.batch_classify(item[0] for item in testData)
#results[:10]
#print(f1_score([item[1] for item in testData], results))

# Logistic Reggression

from sklearn.linear_model import LinearRegression

#linReg = SklearnClassifier(LinearRegression())
#linReg.train((trainData))
#print("Logistic Reggression: ", nltk.classify.accuracy(linReg, testData))

# Ensembles
# Random forest
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

forest = SklearnClassifier(RandomForestClassifier())
forest.train(trainData)
print("Random Forest: ", nltk.classify.accuracy(forest, testData))

adaboost = SklearnClassifier(AdaBoostClassifier())
adaboost.train(trainData)
print("Adaboost: ", nltk.classify.accuracy(adaboost, testData))

#from sklearn.neural_network import

# Support Vector Machines

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

svm = SklearnClassifier(LinearSVC())
svm.train(trainData)

print("SVM: ", nltk.classify.accuracy(svm, testData))

'''
results = svm.batch_classify(item[0] for item in testData)

print(results)

# Compute confusion matrix
import numpy as np

cmm = confusion_matrix([x[1] for x in testData], results)

print(cmm)
cmm = np.array(cmm,dtype = np.float)
print(cmm.shape)

#f=figure()
#ax = f.add_subplot(111)
#show()
#%pylab inline

# Show confusion matrix in a separate window
print(imshow(cmm,interpolation='nearest'),
title('Confusion matrix'),
colorbar(),
ylabel('True label'),
xlabel('Predicted label'))
#pl.
'''

