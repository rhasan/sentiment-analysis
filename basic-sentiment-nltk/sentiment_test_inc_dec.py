from splitter_postagger_nltk import *
from dictionary_tagger import *
from sentiment_score_inc_dec import *

review = """What can I say about this place. The staff of the restaurant is nice and the eggplant is not bad. Apart from that, very uninspired food, lack of atmosphere and too expensive. I am a staunch vegetarian and was sorely dissapointed with the veggie options on the menu. Will be the last time I visit, I recommend others to avoid."""


score = sentiment_score(review)

print "Review: ", review
print "Sentiment: ", score

