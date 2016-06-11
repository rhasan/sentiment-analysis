from splitter_postagger_nltk import *
from dictionary_tagger import *

def value_of(sentiment):
    if sentiment == 'positive': return 1
    if sentiment == 'negative': return -1
    return 0

def sentence_score(sentence):
    total_score = 0.0
    previous_token = None

    for token in sentence:
        tags = token[2]
        token_score = sum([value_of(tag) for tag in tags])
        #print "Debug:", token[0], tags, token_score
        
        if previous_token is not None:
            previous_tags = previous_token[2]
            if 'inc' in previous_tags:
                token_score = token_score * 2
            elif 'dec' in previous_tags:
                token_score = token_score / 2
        total_score = total_score + token_score
        previous_token = token

    return total_score

def sentiment_score(review):
    splitter = Splitter()
    postagger = POSTagger()

    splitted_sentences = splitter.split(review)
    pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

    dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml', 'dicts/inc.yml', 'dicts/dec.yml'])
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)

    return sum([sentence_score(sentence) for sentence in dict_tagged_sentences])
