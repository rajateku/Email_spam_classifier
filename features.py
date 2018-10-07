import re
from collections import Counter
import numpy as np

def create_bag_of_words(filePaths):
    bagOfWords = []
    regex = re.compile("X-Spam.*\n")
    for filePath in filePaths:
        with open(filePath, encoding="latin-1") as f:
            raw = f.read()
            raw = re.sub(regex, '', raw)
            tokens = raw.split()
            for token in tokens:
                bagOfWords.append(token)
    return bagOfWords


def get_feature_matrix(filePaths, featureDict):
    featureMatrix = np.zeros(shape=(len(filePaths),
                                    len(featureDict)),
                             dtype=float)
    regex = re.compile("X-Spam.*\n")
    for i, filePath in enumerate(filePaths):
        with open(filePath, encoding="latin-1") as f:
            _raw = f.read()
            raw = re.sub(regex, '', _raw)
            tokens = raw.split()
            fileUniDist = Counter(tokens)
            for key, value in fileUniDist.items():
                if key in featureDict:
                    featureMatrix[i, featureDict[key]] = value
    return featureMatrix

def regularize_vectors(featureMatrix):
    for doc in range(featureMatrix.shape[0]):
        totalWords = np.sum(featureMatrix[doc, :], axis=0)
        featureMatrix[doc, :] = np.multiply(featureMatrix[doc, :], (1 / (totalWords + 1)))

    return featureMatrix