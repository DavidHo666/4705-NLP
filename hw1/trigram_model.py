import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    res = []
    sequence.insert(0, 'START')
    sequence.append('STOP')

    if n>2:
        for _ in range(n-2):
            sequence.insert(0, 'START')

    for i in range(0, len(sequence) - n + 1):
        res.append(tuple(sequence[i:i+n]))

    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        self.total_words_count = None

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)


        ##Your code here
        for sequence in corpus:
            for item in get_ngrams(sequence, 1):
                self.unigramcounts[item] += 1
            for item in get_ngrams(sequence, 2):
                self.bigramcounts[item] += 1
            for item in get_ngrams(sequence, 3):
                self.trigramcounts[item] += 1


    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0] == trigram[1] == 'START':
            return self.raw_bigram_probability(trigram[-2:])

        if self.bigramcounts[trigram[0:2]] == 0:
            return 1/len(self.lexicon)
        else:
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[0:2]]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[bigram[0:1]] == 0:
            return 1/len(self.lexicon)
        else:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[0:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        if self.total_words_count is None:
            self.total_words_count = sum(self.unigramcounts.values()) - self.unigramcounts[('START', )]

        return self.unigramcounts[unigram]/self.total_words_count

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        cur = (None, 'START', 'START')
        res = []
        i = 0

        while cur[2] != 'STOP' and i < t:
            first, second = cur[1], cur[2]
            matched = [item for item in self.trigramcounts.keys() if (item[0] == first and item[1] == second)]
            if ('START', 'START', 'START') in matched:
                matched.remove(('START', 'START', 'START'))
            probs = [self.raw_trigram_probability(item) for item in matched]
            r = random.random()
            prob_sum = 0
            for j in range(len(probs)):
                prob_sum += probs[j]
                if prob_sum > r:
                    cur = matched[j]
                    res.append(cur)
                    break
        return res

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        return (lambda1 * self.raw_trigram_probability(trigram)
                + lambda2 * self.raw_bigram_probability(trigram[-2:])
                + lambda3 * self.raw_unigram_probability(trigram[-1:]))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        res = 0
        for trigram in trigrams:
            res += math.log2(self.smoothed_trigram_probability(trigram))

        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total_words = 0
        the_sum = 0
        for sentence in corpus:
            total_words += len(sentence)+1 # adding one STOP token at the end
            the_sum += self.sentence_logprob(sentence)
        return 2**(-the_sum/total_words)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += 1 if pp_1 < pp_2 else 0
    
        for f in os.listdir(testdir2):
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            correct += 1 if pp_2 < pp_1 else 0
        return correct/total

if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1])
    model = TrigramModel('hw1_data/brown_train.txt')
    # print(model.generate_sentence())

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # dev_corpus = corpus_reader('hw1_data/brown_test.txt', model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt',
                                   'hw1_data/ets_toefl_data/train_low.txt',
                                   'hw1_data/ets_toefl_data/test_high',
                                   'hw1_data/ets_toefl_data/test_low')
    print(acc)

