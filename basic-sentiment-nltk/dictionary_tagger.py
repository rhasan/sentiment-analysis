import yaml

class DictionaryTagger(object):
    def __init__(self, dictionayr_paths):
       files = [open(path,'r') for path in dictionayr_paths]
       dictionaries = [yaml.load(dict_file) for dict_file in files]
       map(lambda x: x.close(), files)
       self.dictionary = {}
       self.max_key_size = 0
       for curr_dict in dictionaries:
           for key in curr_dict:
               if key in self.dictionary:
                   self.dictionary.extend(curr_dict[key])
               else:
                   self.dictionary[key] = curr_dict[key]
                   self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemma=False):
        tag_sentence = []
        N = len(sentence)

        if self.max_key_size == 0:
            self.max_key_size = N

        i = 0

        while (i < N):
            j = min(i + self.max_key_size, N)
            tagged = False

            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                if tag_with_lemma:
                    literal = expression_lemma
                else:
                    literal = expression_form

                if literal in self.dictionary:
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i = i + 1
        return tag_sentence
