import itertools

from nltk.classify import ClassifierI
from nltk.classify import NaiveBayesClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import NuSVC


class MaxVoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self._labels = sorted(set(itertools.chain(*[c.labels() for c in classifiers])))

    def labels(self):
        return self._labels

    def classify(self, feats):
        counts = FreqDist()
        for classifier in self._classifiers:
            counts[classifier.classify(feats)] += 1

        return counts.max()

def trainClassifier(train_feats):
    print("Training NB...")
    nb_classifier = NaiveBayesClassifier.train(train_feats)

    print("Training NuSVC...")
    svc_classifier = SklearnClassifier(NuSVC()).train(train_feats)

    print("Training MultinomialNB...")
    mnb_classifier = SklearnClassifier(MultinomialNB()).train(train_feats)

    mv_classifier = MaxVoteClassifier(nb_classifier, svc_classifier, mnb_classifier)

    return mv_classifier