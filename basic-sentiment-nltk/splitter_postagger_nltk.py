import nltk

class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        Splits a text into a list sentences.
        A sentence is represented as a list of words.
        E.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]

        @param text: a text.
        @return: a list of lists of words. 
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences

class POSTagger(object):
    def __init__(self):
        pass

    def pos_tag(self, sentences):
        """
        Takes a list of lists of words representing a list of sentences
        and returns a list of lists of taggend tokens where each tagged
        token has the form of original word, lemma, and a list of pos tags.
        E.g. input: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        E.g. of returnd value: [[('this', 'this', ['DT']), ('is', 'be', ['VB']), ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
              [('this', 'this', ['DT']), ('is', 'be', ['VB']), ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        @param sentences: list of lists of words.
        @return: list of lists of tagged tokens.
        """

        pos = [nltk.pos_tag(sent) for sent in sentences]
        pos_final = [[(word, word, [postag]) for (word, postag) in sentence] for sentence in pos]
        return pos_final
