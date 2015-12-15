__author__ = 'CJ'

import nltk
import random
from collections import defaultdict
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

vectorizer = TfidfVectorizer()

newsgroups = nltk.corpus.PlaintextCorpusReader('./data/', '.*/[0-9]+', encoding='latin1')
#newsgroups = vectorizer.fit_transform(newsgroups)
ids = newsgroups.fileids()

random.seed(0)
random.shuffle(ids)
#ids = ids[:5000]  # Change for whole dataset
size = len(ids)

testSet = ids[:int(size*0.1)]
trainSet = ids[int(size*0.1):]

print("Length of training data: ", len(trainSet), "\nLength of test data: ", len(testSet))

def vectorize(text):
    vectorizer = feature_extraction.text.CountVectorizer(
        stop_words='english',
        ngram_range=(1, 1),  #ngram_range=(1, 1) is the default
        dtype='double',
        )
    #text = [w.lower() for w in vectorizer]
    return vectorizer.fit_transform(text)

#print(vectorize(trainSet))

def features(text):
    # Convert a post into a dictionary of features
    features = defaultdict(int)
    for word in text:  # should just be text instead of vectorizer
        if word.isalpha():
            features[word.lower()] += 1
    return features


def getclass(fileid):
    # Returns class from fileid
    return fileid.split('/')[0]

# Set training data and testData
trainData = [(features(newsgroups.words(fileids=f)),getclass(f)) for f in trainSet]
testData = [(features(newsgroups.words(fileids=f)),getclass(f)) for f in testSet]

# For vectorisation
#trainData = [(vectorize(newsgroups.words(fileids=f)),getclass(f)) for f in trainSet]
#testData = [(vectorize(newsgroups.words(fileids=f)),getclass(f)) for f in testSet]



#c = nltk.FreqDist(item[1] for item in trainData)
#default = c.keys()[0]
#default = (next(iter(c.values())))

#print("c: ", c)

#sum(c == default for f, c in testData) / float(len(testData))


def main():
    # Naive Bayes
    nb = nltk.NaiveBayesClassifier.train(trainData)
    print("NB: ", nltk.classify.accuracy(nb, testData))

    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    # BernoulliNB
    bernoulli = SklearnClassifier(BernoulliNB())
    bernoulli.train(trainData)
    print("NB Bernoulli: ", nltk.classify.accuracy(bernoulli, testData))
    #gaussian = SklearnClassifier(GaussianNB())
    #gaussian.train(trainData.toarray(trainData))
    #print("Gaussian: ", nltk.classify.accuracy(gaussian, testData))

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
    #print(results[:10])
    #print(f1_score([item[1] for item in testData], results))

    # Logistic Reggression

    from sklearn.linear_model import LinearRegression

    #linReg = SklearnClassifier(LinearRegression())
    #linReg.train((trainData))
    #print("Logistic Reggression: ", nltk.classify.accuracy(linReg, testData))

    # Ensembles
    # Random forest
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    forest = SklearnClassifier(RandomForestClassifier(n_estimators=100))
    forest.train(trainData)
    print("Random Forest: ", nltk.classify.accuracy(forest, testData))

    # AdaBoost
    adaboost = SklearnClassifier(AdaBoostClassifier())
    adaboost.train(trainData)
    print("Adaboost: ", nltk.classify.accuracy(adaboost, testData))


    # Support Vector Machines

    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix

    svm = SklearnClassifier(LinearSVC(loss="hinge"))
    svm.train(trainData)

    print("SVM: ", nltk.classify.accuracy(svm, testData))


    # KMeans

    from sklearn.cluster import KMeans
    km = SklearnClassifier(KMeans())
    km.train(trainData)
    print("KMeans: ", nltk.classify.accuracy(km, testData))

    # K nearest neighbors
    #from sklearn.neighbors import KNeighborsClassifier
    #knn = SklearnClassifier(KNeighborsClassifier)
    #knn.train(trainData)
    #print("KNN: ", nltk.classify.accuracy(knn, testData))

'''
# Some tests with other ways as representation
def main2():
    print("Using another representation: ")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    train = vectorizer.fit_transform(trainSet)
    test = vectorizer.fit_transform(trainSet)
    print("Data is loaded")

    # SVM

'''


def main3():
    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot

    svm = SklearnClassifier(LinearSVC(loss="hinge"))
    svm.train(trainData)
    print("SVM: ", nltk.classify.accuracy(svm, testData))
    results = svm.classify_many(item[0] for item in testData)

    print(results)
    from sklearn.metrics import classification_report

    # getting a full report
    print(classification_report(testData, results))

    # Compute confusion matrix
    import numpy as np
    cmm = confusion_matrix([x[1] for x in testData], results)

    print(cmm)
    cmm = np.array(cmm, dtype = np.float)
    print(cmm.shape)

    #f=figure()
    #ax = f.add_subplot(111)
    #show()
    #%pylab inline

    # Show confusion matrix in a separate window
    print(pyplot.imshow(cmm, interpolation='nearest'))
    #pl.



if __name__ == '__main__':
    main()
    #main2()
    #main3()





