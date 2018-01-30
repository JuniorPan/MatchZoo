#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'haolanchen'

from nltk import word_tokenize
from gensim.models.phrases import *
import pickle as pkl
import string

sys.path.append('..')

from utils.extra_config import *
from utils.logger import *

class PhraseExtractor(Logger):
    '''
    extract phrases from a sentence
    '''
    def __init__(self):
        self.MODEL_PKL = wkdp('phrase_extractor.pkl')
        self.WK_CORPUS = WK_CORPUS
        self.bi_phraser = None
        self.tri_phraser = None
        self.THRESHOLD = 5.0
        self.MIN_COUNT = 5
        super(PhraseExtractor, self).__init__()

    def build(self, corpus_path=WK_CORPUS):
        '''
        load or build extractor model
        :param corpus_path:
        :return:
        '''
        if load_phrase_extractor:
            with open(self.MODEL_PKL, 'rb') as f:
                self.bi_phraser = pkl.load(f)
                self.tri_phraser = pkl.load(f)
        else:
            with open(corpus_path) as corpus_file:
                term_mat = PhraseExtractor.get_term_mat(corpus_file)
                bi_phrases = Phrases(term_mat, min_count=self.MIN_COUNT, threshold=self.THRESHOLD)
                self.bi_phraser = Phraser(bi_phrases)
                tri_phrases = Phrases(self.bi_phraser[term_mat], min_count=self.MIN_COUNT, threshold=self.THRESHOLD)
                self.tri_phraser = Phraser(tri_phrases)
                with open(self.MODEL_PKL, 'wb') as f:
                    pkl.dump(self.bi_phraser, f)
                    pkl.dump(self.tri_phraser, f)
        return self

    def extract(self, sent, limit=3):
        '''
        extract phrases from input sent
        :param sent:
        :param limit: bigram or trigram
        :return: List[ _ concat phrase ]
        '''
        terms = PhraseExtractor.get_sent_terms(sent)
        if limit >= 2:
            terms = self.bi_phraser[terms]
        if limit >= 3:
            terms = self.tri_phraser[terms]
        return terms

    @staticmethod
    def get_sent_terms(s):
        '''
        extract terms from a sent without case or punct
        :param s: str
        :return: List[Str]
        '''
        clean_sent = s.lower().translate({ord(k): None for k in string.punctuation})
        tokens = word_tokenize(clean_sent)
        return tokens

    @staticmethod
    def get_term_mat(corpus_file):
        '''
        get term mat from wikiqa corpus file
        :param corpus_file: wikiqa corpus
        :return: List[List[Str]]
        '''
        term_mat = []
        for line in corpus_file:
            try:
                term_mat.append(PhraseExtractor.get_sent_terms(line)[1:])
            except:
                log.error('Error when parsing: {}'.format(line))
        return term_mat

def unittest():
    extractor = PhraseExtractor().build()
    text = 'D1222 The Australian Recording Industry Association Music Awards ( commonly known as ARIA Music Awards\ ' \
           ' or ARIA Awards ) is an annual series of awards nights celebrating the Australian music industry , put on\ ' \
           ' by the Australian Recording Industry Association ( ARIA ) .'
    biphrase = extractor.extract(text,2)
    triphrase = extractor.extract(text,3)
    print('biphrase: {}'.format(' '.join(biphrase)))
    print('triphrase: {}'.format(' '.join(triphrase)))

if __name__ == '__main__':
    unittest()