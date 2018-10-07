import glob
import random
from collections import Counter
import numpy as np

from features import create_bag_of_words
from features import get_feature_matrix
from features import regularize_vectors

hamFolder = "/home/raja/Desktop/Email_spam_classifier/data/ham"
spamFolder = "/home/raja/Desktop/Email_spam_classifier/data/spam"
percentTest = .1

def data_generation(hamFolder, spamFolder, percentTest):
    pathLabelPairs = {}
    for hamPath in glob.glob(hamFolder + '/*'):
        pathLabelPairs.update({hamPath: "0.,1."})
    for spamPath in glob.glob(spamFolder + '/*'):
        pathLabelPairs.update({spamPath: "1.,0."})

    numTest = int(percentTest * len(pathLabelPairs))
    print(numTest)
    print(pathLabelPairs.items())
    testing = set(random.sample(pathLabelPairs.items(), numTest))

    for entry in testing:
        del pathLabelPairs[entry[0]]

    trainPaths = []
    trainY = []
    for item in pathLabelPairs.items():
        trainPaths.append(item[0])
        trainY.append([float(i) for i in item[1].split(',')])
    del pathLabelPairs
    trainY = np.asarray(trainY)

    testPaths = []
    testY = []
    for item in testing:
        testPaths.append(item[0])
        testY.append([float(i) for i in item[1].split(',')])
    del testing
    testY = np.asarray(testY)

    bagOfWords = create_bag_of_words(trainPaths)

    k = 5
    freqDist = Counter(bagOfWords)
    newBagOfWords = []
    for word, freq in freqDist.items():
        if freq > k:
            newBagOfWords.append(word)
    features = set(newBagOfWords)
    featureDict = {feature: i for i, feature in enumerate(features)}

    trainX = get_feature_matrix(trainPaths, featureDict)
    testX = get_feature_matrix(testPaths, featureDict)

    trainX = regularize_vectors(trainX)
    testX = regularize_vectors(testX)

    return trainX, trainY, testX, testY


X,Y,x,y = data_generation(hamFolder,spamFolder,percentTest)
print(x,y)
