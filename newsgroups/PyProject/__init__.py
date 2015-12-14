__author__ = 'CJ'

import os, codecs, math

def __init__(self, trainingDir, **stopwords):
    self.vocabulary = {}
    self.prob = {}
    self.totals = {}
    self.stopwords = {}
    from nltk.corpus import stopwords
    f = open(stopwords.words("english")) #open(stopwords)
    for line in f:
        self.stopwords[line.strip()] = 1
    f.close()
    categories = os.listdir(trainingDir)
    #filter out files that are not directories
    self.categories = [filename for filename in categories
    if os.path.isdir(trainingDir + filename)]
    print("Counting ...")
    for category in self.categories:
        print(' ' + category)
        (self.prob[category],
        self.totals[category]) = self.train(trainingDir, category)
    # I am going to eliminate any word in the vocabulary
    # that doesn't occur at least 3 times
    toDelete = []
    for word in self.vocabulary:
        if self.vocabulary[word] < 3:
            # mark word for deletion
            # can't delete now because you can't delete
            # from a list you are currently iterating over
            toDelete.append(word)
        # now delete
        for word in toDelete:
            del self.vocabulary[word]
        # now compute probabilities
        vocabLength = len(self.vocabulary)
        print("Computing probabilities:")
        for category in self.categories:
            print(' ' + category)
            denominator = self.totals[category] + vocabLength
            for word in self.vocabulary:
                if word in self.prob[category]:
                    count = self.prob[category][word]
                else:
                    count = 1
                self.prob[category][word] = (float(count + 1)
                                             / denominator)
        print("DONE TRAINING\n\n")

def train(self, trainingdir, category):
    """counts word occurrences for a particular category"""
    currentdir = trainingdir + category
    files = os.listdir(currentdir)
    counts = {}
    total = 0
    for file in files:#print(currentdir + '/' + file)
        f = codecs.open(currentdir + '/' + file, 'r', 'iso8859-1')
        for line in f:
            tokens = line.split()
            for token in tokens:
            # get rid of punctuation and lowercase token
                token = token.strip('\'".,?:-')
                token = token.lower()
                if token != '' and not token in self.stopwords:
                    self.vocabulary.setdefault(token, 0)
                    self.vocabulary[token] += 1
                    counts.setdefault(token, 0)
                    counts[token] += 1
                    total += 1
        f.close()
    return(counts, total)

