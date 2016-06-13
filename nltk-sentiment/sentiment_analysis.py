from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize

class SentimentAnalyzerTry(object):
    def __init__(self):
        self.n_instances = 1000
        self.n_training = int(self.n_instances * 0.8)
        self.n_testing = int(self.n_instances * 0.2)
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
        training_end = self.n_training
        testing_start = training_end
        testing_end = testing_start + self.n_testing

        
        train_subj_docs = subj_docs[:training_end]
        test_subj_docs = subj_docs[testing_start:testing_end]

        train_obj_docs = obj_docs[:training_end]
        test_obj_docs = obj_docs[testing_start:testing_end]
        
        self.training_docs = train_subj_docs + train_obj_docs
        self.testing_docs = test_subj_docs + test_obj_docs
        

    def extract_training_test_features(self):
        # We use simple unigram word features, handling negation.
        self.all_words_neg = self.mark_negative_sentence(self.training_docs)
        self.unigram_feats = self.sentim_analyzer.unigram_word_feats(self.all_words_neg, min_freq=4)
        self.sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=self.unigram_feats)

        # We apply features to obtain a feature-value representation of our datasets.
        self.training_set = self.sentim_analyzer.apply_features(self.training_docs)
        self.test_set = self.sentim_analyzer.apply_features(self.testing_docs)
        
    def mark_negative_sentence(self, docs):
        all_words_neg = self.sentim_analyzer.all_words([mark_negation(doc) for doc in docs])
        return all_words_neg

    def train_sentiment_analyzer(self, evaluate=True):
        self.prepare_training_and_test_data()
        self.extract_training_test_features()

        # We can now train our classifier on the training set, and subsequently output the evaluation results
        self.trainer = NaiveBayesClassifier.train
        self.classifier = self.sentim_analyzer.train(self.trainer, self.training_set)

        if evaluate:
            self.evaluate_classifier()

    def evaluate_classifier(self):
        for key, value in sorted(self.sentim_analyzer.evaluate(self.test_set).items()):
            print('{0}: {1}'.format(key, value))

    def classify_text(self, text):
        self.sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=self.unigram_feats)
        return self.classifier.classify(self.sentim_analyzer.extract_features(tokenize.word_tokenize(text)))


def main():
    print "Sentiment Analysis experiment -- Subjective/Objective classifier"
    sent_classifier = SentimentAnalyzerTry()
    sent_classifier.train_sentiment_analyzer()

    obj_example = "the train went to dublin where a young man got into the train"
    print "Text: ", obj_example
    print sent_classifier.classify_text(obj_example)
    
    subj_example = "the journey by train was a nice experience. it was long but enjoyable"
    print "Text: ", subj_example
    print sent_classifier.classify_text(subj_example)

if __name__ == '__main__':
    main()

