from splitter_postagger_nltk import *
from dictionary_tagger import *

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentiment_score(review):
    splitter = Splitter()
    postagger = POSTagger()

    splitted_sentences = splitter.split(review)
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml'])
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

    return sum([value_of(tag) for sentence in dict_tagged_sentences for tokens in sentence for tag in tokens[2]])

