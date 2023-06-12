#!/usr/bin/env python
import string
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List
from collections import defaultdict

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    res = set()
    for synset in wn.synsets(lemma, pos):
        for l in synset.lemmas():
            if l.name() != lemma:
                res.add(l.name().replace("_", " "))
    return list(res)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    map = defaultdict(lambda: 0)
    for synset in wn.synsets(context.lemma, context.pos):
        for lemma in synset.lemmas():
            if lemma.name() != context.lemma:
                map[lemma.name().replace("_", " ")] += lemma.count()
    return max(map, key=map.get)

def wn_simple_lesk_predictor(context : Context) -> str:
    stop_words = stopwords.words('english')
    map = dict()
    max_count = 0
    most_frequent_res = None
    for k, synset in enumerate(wn.synsets(context.lemma, context.pos)):
        definitions_examples = []
        definitions_examples += [w for w in tokenize(synset.definition()) if w not in stop_words]
        # definitions_examples += synset.lemma_names()
        definitions_examples += [w for example in synset.examples() for w in tokenize(example) if w not in stop_words]

        for hypernym in synset.hypernyms():
            definitions_examples += [w for w in tokenize(hypernym.definition()) if w not in stop_words]
            # definitions_examples += hypernym.lemma_names()
            definitions_examples += [w for example in hypernym.examples()
                                     for w in tokenize(example) if w not in stop_words]

        map[k] = len(set(definitions_examples) &
                               set([w for w in context.left_context if w not in stop_words] +
                                   [w for w in context.right_context if w not in stop_words]))

        cur_count = 0
        for lemma in synset.lemmas():
            cur_count += lemma.count()
        if cur_count >= max_count and len(synset.lemmas()) > 1:
            max_count = cur_count
            most_frequent_res = synset

    most_overlap_synsets = [wn.synsets(context.lemma, context.pos)[k] for k, v in map.items() if v == max(map.values())]

    max_count = 0
    most_frequent_synset_in_overlape = None
    for s in most_overlap_synsets:
        cur_count = 0
        for l in s.lemmas():
            cur_count += l.count()
        if cur_count >= max_count and len(s.lemmas()) > 1:
            max_count = cur_count
            most_frequent_synset_in_overlape = s
    # most_frequent_synset_in_overlape = max(most_overlap_synsets, key=most_overlap_synsets.count)
    # most_frequent_lexeme_from_most_frequent_synset = max(most_frequent_synset_in_overlape.lemmas(),
    #                                                      key=most_frequent_synset_in_overlape.lemmas().count)
    max_count = 0
    res = None
    if most_frequent_synset_in_overlape is None:
        for l in most_frequent_res.lemmas():
            if l.count() >= max_count and l.name() != context.lemma:
                max_count = l.count()
                res = l
    else:
        for l in most_frequent_synset_in_overlape.lemmas():
            if l.count() >= max_count and l.name() != context.lemma:
                max_count = l.count()
                res = l

    return res.name().replace("_", " ")



class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        best_similarity = 0
        best_candidate = None
        for candidate in candidates:
            try:
                if self.model.similarity(context.lemma, candidate) > best_similarity:
                    best_similarity = self.model.similarity(context.lemma, candidate)
                    best_candidate = candidate
            except KeyError:
                continue

        return best_candidate


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = get_candidates(context.lemma, context.pos)
        idx = len(context.left_context) + 1
        input_toks = self.tokenizer.encode(context.left_context + ['[MASK]'] + context.right_context)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx])[::-1]
        best_word = None
        for w in self.tokenizer.convert_ids_to_tokens(best_words):
            if w in candidates:
                best_word = w
                break
        return best_word

class MyBertPredictor(BertPredictor):
    # def __int__(self, filename):
    #     self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    #     self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #     self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        candidates = self.__my_get_candidates(context.lemma, context.pos)
        stop_words = stopwords.words('english')
        left_tokens = [w for w in context.left_context]
        right_tokens = [w for w in context.right_context]
        idx = len(left_tokens) + 1
        input_toks = self.tokenizer.encode(left_tokens + ['[MASK]'] + right_tokens)
        # tmp_token = self.tokenizer.convert_ids_to_tokens(input_toks)
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][idx])[::-1]
        best_word = None
        for w in self.tokenizer.convert_ids_to_tokens(best_words):
            if w in candidates:
                best_word = w
                break
        return best_word

    def __my_get_candidates(self, lemma, pos):
        res = set()
        for synset in wn.synsets(lemma, pos):
            for l in synset.lemmas():
                if l.name() != lemma:
                    res.add(l.name().replace("_", " "))

            for hypernym in synset.hypernyms():
                for l1 in hypernym.lemmas():
                    if l1.name() != lemma:
                        res.add(l1.name().replace("_", " "))

            for hyponym in synset.hyponyms():
                for l2 in hyponym.lemmas():
                    if l2.name() != lemma:
                        res.add(l2.name().replace("_", " "))

        return list(res)

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    # W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    # predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        # prediction = smurf_predictor(context)
        # prediction = wn_frequency_predictor(context)
        # prediction = wn_simple_lesk_predictor(context)
        # prediction = predictor.predict_nearest(context)
        # bert_predictor = BertPredictor()
        # prediction = bert_predictor.predict(context)
        my_bert_predictor = MyBertPredictor()
        prediction = my_bert_predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
