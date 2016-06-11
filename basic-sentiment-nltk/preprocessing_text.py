from splitter_postagger_nltk import *
from dictionary_tagger import *
from pprint import pprint

text = """What can I say about this place. The staff of the restaurant is nice and the eggplant is not bad. Apart from that, very uninspired food, lack of atmosphere and too expensive. I am a staunch vegetarian and was sorely dissapointed with the veggie options on the menu. Will be the last time I visit, I recommend others to avoid."""

splitter = Splitter()
postagger = POSTagger()

splitted_sentences = splitter.split(text)
print splitted_sentences

pos_tagged_sentences = postagger.pos_tag(splitted_sentences)

print pos_tagged_sentences

dicttagger = DictionaryTagger([ 'dicts/positive.yml', 'dicts/negative.yml'])
dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
pprint(dict_tagged_sentences)


