#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'haolanchen'

import os
import sys

pwd = os.getcwd()
pj = os.path.join

MZ_PATH = '/usr/local/app/ceph1/MatchZoo/'
MZ_CONF_PATH = pj(MZ_PATH, 'matchzoo/models/wikiqa_config/')
WIKIQA_DATA_PATH = pj(MZ_PATH, 'data/WikiQA/')

def confp(token):
    '''
    combine absolute parent path of MatchZoo conf and filenames
    @param token: filename
    @return: joint path
    '''
    return pj(MZ_CONF_PATH, token)

def wkdp(token):
    '''
    combine absolute parent path of wikiqa data and filenames
    @param token: filename
    @return: joint path
    '''
    return pj(WIKIQA_DATA_PATH, token)

WORD_DICT_PATH = wkdp('word_dict.txt')
GEMINET_CONF_PATH = confp('geminet_wikiqa.config')

WK_CORPUS = wkdp('corpus.txt')

load_phrase_extractor = True