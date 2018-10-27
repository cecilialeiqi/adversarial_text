import copy
from math import log
from collections import Counter

import nltk
from nltk import word_tokenize
from nltk.util import ngrams

"""
P(c|b, a) = C(a, b, c) / C(a, b)
"""

class NGramLangModel(object):
    def __init__(self, corpus, n):
        self.default = 0.0001
        self.n = n
        self.ngram_counter = [None] * n
        for i in range(n, 0, -1):
            self.ngram_counter[i - 1] = NGramLangModel.get_ngram(corpus, i)

    @staticmethod
    def get_ngram(corpus, n):
        counter = Counter()
        for doc in corpus:
            counter.update(ngrams(doc, n))
        return counter

    def log_prob(self, x, n):
        if x[0] is None:
            return self.log_prob(x[1:], n - 1)
        elif x[-1] is None:
            return self.log_prob(x[:-1], n - 1)
        else:
            logp = log(self.ngram_counter[n - 1].get(tuple(x), self.default))
            #print(logp)
            logp -= log(self.ngram_counter[n - 2].get(tuple(x[:-1]), self.default))
            #print(',',logp)
            return logp

    def log_prob_diff(self, sentence, pos, repl):
        pad = [None] * (self.n - 1)
        padded_sent = pad + sentence + pad

        window_words = padded_sent[pos:pos + 2 * self.n - 1]
        repl_window_words = copy.copy(window_words)
        repl_window_words[self.n - 1] = repl

        log_p = 0
        repl_log_p = 0
        for i in range(self.n, 2 * self.n):
            repl_log_p += self.log_prob(repl_window_words[i - self.n:i], self.n)
            log_p += self.log_prob(window_words[i - self.n:i], self.n)

        return (log_p - repl_log_p)/self.n
    def log_prob_alone(self,sentence):
        pad = [None] * (self.n - 1)
        padded_sent = pad + sentence + pad
        log_p = 0
        for pos in range(len(sentence)):
            window_words = padded_sent[pos:pos + 2 * self.n - 1]
            for i in range(self.n, 2 * self.n):
                log_p += self.log_prob(window_words[i - self.n:i], self.n)
        if len(sentence):
            return log_p/len(sentence)
        else:
            return 0
if __name__ == '__main__':
    text = [word_tokenize("I need to write a program in NLTK that breaks a corpus (a large collection of \
            txt files) into unigrams, bigrams, trigrams, fourgrams and fivegrams. \
            I have to write a program in NLTK that breaks a corpus \
            I have to write a program in spacy that breaks a corpus (a large collection of \
            txt files) into unigrams, bigrams, trigrams, fourgrams and fivegrams. \
            I have to write a program in NLTK that breaks a corpus")]

    ngram = NGramLangModel(text, 3)
    print(ngram.log_prob_diff(text[0], 1, 'have'))
