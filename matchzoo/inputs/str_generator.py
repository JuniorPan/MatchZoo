#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'haolanchen'

import numpy as np
import json
import sys

sys.path.append('..')

from preprocess import Preprocess
from layers import DynamicMaxPooling
from utils.extra_config import *
from utils.logger import *


def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[0]] = int(r[1])
    return word_dict

class InputTransformer(Logger):
    '''
    transform original term lists to matchzoo inputs
    '''
    def __init__(self, config):
        self.config = config['inputs']['share']
        self._word_seg_config = {'enable': True, 'lang': 'en'}
        self.fill_word = self.config['vocab_size'] - 1
        super(InputTransformer, self).__init__()

    def build(self):
        self.word_dict = read_dict(WORD_DICT_PATH)
        log.info('built word_dict')
        return self

    def unify_terms(self,_term_list):
        '''
        unify terms in the term list
        @param _term_mat: List[Str]
        @return: List[Str]
        '''
        _term_list = Preprocess.word_seg([_term_list,],self._word_seg_config)[0] # fit in input type of [[term]]
        _term_list = Preprocess.word_lower([_term_list,])[0]
        return _term_list

    def get_raw_index(self, _term_list):
        '''
        get raw id arr from term list
        @param _term_list:List[Str] seg results of sentence
        @return: List[Int]
        '''
        _index_list = [self.word_dict[_w] for _w in _term_list if _w in self.word_dict]
        return _index_list

    def get_index_len(self, term_list, max_len):
        '''
        get clean input from raw id arr
        @param _term_list:List[Int] seg results of sentence
        @return: np.array(np.array(index)), np.array(Int)
        '''
        index_list = self.get_raw_index(term_list)[:max_len]
        index_len = len(index_list)
        index_arr = np.array([self.fill_word for i in range(max_len)])
        for i, index in enumerate(index_list):
            index_arr[i] = index
        return np.array([index_arr,]), np.array([index_len,])

    def get_dpool_index(self, _len1, _len2):
        '''
        get dynamic pooling index
        @param _len1: Int length of text1 terms
        @param _len2: Int length of text2 terms
        @return: np.array[Int]
        '''
        _dpool_index_arr = DynamicMaxPooling.dynamic_pooling_index([_len1,], [_len2,], self.config['text1_maxlen'], self.config['text2_maxlen'])
        return _dpool_index_arr

    def genr_input(self, term_list1, term_list2):
        '''
        aggregate results
        @param term_list1: List[Str]
        @param term_list2: List[Str]
        @return: dict[Str,Obj]
        '''
        term_list1 = self.unify_terms(term_list1)
        log.info('term list 1: {}'.format(' '.join(term_list1)))
        index_arr1,len_arr1 = self.get_index_len(term_list1, self.config['text1_maxlen'])

        term_list2 = self.unify_terms(term_list2)
        log.info('term list 2: {}'.format(' '.join(term_list2)))
        index_arr2,len_arr2 = self.get_index_len(term_list2, self.config['text2_maxlen'])

        model_input = {'query':index_arr1, 'query_len':len_arr1,\
                       'doc':index_arr2, 'doc_len':len_arr2}

        if self.config['use_dpool']:
            dpool_index = self.get_dpool_index(len_arr1, len_arr2)
            model_input['dpool_index'] = dpool_index

        return model_input

    def unittest(self):
        '''
        test the procedure from original term list to model input dict
        @return:
        '''
        term_list1 = 'Double dopaminergic thistle'
        term_list2 = 'completes herbs textures topical'
        self.build()

        model_input = self.genr_input(term_list1, term_list2)
        print(model_input)

if __name__ == '__main__':
    with open(GEMINET_CONF_PATH, 'r') as f:
        config = json.load(f)
    transformer = InputTransformer(config)
    transformer.unittest()