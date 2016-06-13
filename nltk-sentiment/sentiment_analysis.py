from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

class SentimentAnalyzerTry(object):
    def __init__(self):
        self.n_instances = 100
        self.sentim_analyzer = SentimentAnalyzer()

    def prepare_training_and_test_data(self):
        """
        Each document is represented by a tuple (sentence, label). The sentence is tokenized, so it is represented by a list of strings.
        E.g: (['smart', 'and', 'alert', ',', 'thirteen', 'conversations', 'about', 'one',
              'thing', 'is', 'a', 'small', 'gem', '.'], 'subj')
        """
        subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:self.n_instances]]
        obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:self.n_instances]]

        # We separately split subjective and objective instances to keep a balanced uniform class distribution in both train and test sets.
        train_subj_docs = subj_docs[:80]
        test_subj_docs = subj_docs[80:100]

        train_obj_docs = obj_docs[:80]
        test_obj_docs = obj_docs[80:100]
        self.training_docs = train_subj_docs + train_obj_docs
        self.testing_docs = test_subj_docs + test_obj_docs
        

    def extract_training_test_features(self):
        # We use simple unigram word features, handling negation.
        self.all_words_neg = mark_negative_sentence(self.training_docs)
        self.unigram_feats = self.sentim_analyzer.unigram_word_feats(self.all_words_neg, min_freq=4)
        self.sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=self.unigram_feats)

        # We apply features to obtain a feature-value representation of our datasets.
        self.training_set = self.sentim_analyzer.apply_features(self.training_docs)
        self.test_set = self.sentim_analyzer.apply_features(self.testing_docs)
        
    def mark_negative_sentence(self, docs):
        all_words_neg = self.sentim_analyzer.all_words([mark_negation(doc) for doc in docs])
        return all_words_neg

    def train_sentiment_analyzer(self):
        prepare_training_and_test_data()
        extract_training_test_features()

        # We can now train our classifier on the training set, and subsequently output the evaluation results
        self.trainer = NaiveBayesClassifier.train
        self.classifier = self.sentim_analyzer.train(self.trainer, self.training_set)

        evaluate_classifier()

    def evaluate_classifier(self):
        for key, value in sorted(self.sentim_analyzer.evaluate(self.test_set).items()):
            print('{0}: {1}'.format(key, value))

    def classify_text(self, text):
        pass


def main():
    print "Hello world!"

if __name__ == '__main__':
    main()

