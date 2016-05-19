from nltk.classify.util import accuracy
from nltk.corpus import movie_reviews

import zaCode.ClassifierTrainer as ClassifierTrainer
import zaCode.Toolbox as Toolbox


def makePrediction():

    labels = movie_reviews.categories()
    print("Labels for reviews are: {}\n".format(labels) )

    labeled_words = [(label, movie_reviews.words(categories=[label])) for label in labels]
    print("Labeled words:{}\n".format(labeled_words[:10]))

    high_info_words = set(Toolbox.high_information_words(labeled_words))
    print("High information  words:{}\n".format(list(high_info_words)[:10]))

    feat_det = lambda words: Toolbox.bag_of_words_in_set(words, high_info_words)

    lfeats = Toolbox.label_feats_from_corpus(movie_reviews, feature_detector=feat_det)

    train_feats, test_feats = Toolbox.split_label_feats(lfeats)

    mv_classifier = ClassifierTrainer.trainClassifier(train_feats)

    accuracyScore = accuracy(mv_classifier, test_feats)

    print("Accuracy is {}".format(accuracyScore))

if __name__ == '__main__':

    makePrediction()


