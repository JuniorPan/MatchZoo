#!/usr/bin/env python
# coding: utf-8


import os
import sys
sys.path.append('../../matchzoo/')
sys.path.append('../../matchzoo/inputs/')
sys.path.append('../../matchzoo/utils/')

from grammer_parser import tokens_ner, get_ner_postag_list
from preparation import Preparation
from preprocess import Preprocess, NgramUtil

def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict
def read_doc(infile):
    doc = {}
    for line in open(infile):
        r = line.strip().split()
        doc[r[0]] = r[1:]
        #assert len(doc[r[0]]) == int(r[1])
    return doc
def filter_triletter(tri_stats, min_filter_num=5, max_filter_num=10000):
    tri_dict = {}
    tri_stats = sorted(tri_stats.items(), key=lambda d:d[1], reverse=True)
    for triinfo in tri_stats:
        if triinfo[1] >= min_filter_num and triinfo[1] <= max_filter_num:
            if triinfo[0] not in tri_dict:
                tri_dict[triinfo[0]] = len(tri_dict)
    return tri_dict

if __name__ == '__main__':
    prepare = Preparation()
    srcdir = './'
    dstdir = './'

    infiles = [ srcdir + 'WikiQA-mz-train.txt', srcdir + 'WikiQA-mz-dev.txt', srcdir + 'WikiQA-mz-test.txt']
    corpus, rel_train, rel_valid, rel_test = prepare.run_with_train_valid_test_corpus(infiles[0], infiles[1], infiles[2])
    print('total corpus : %d ...' % (len(corpus)))
    print('total relation-train : %d ...' % (len(rel_train)))
    print('total relation-valid : %d ...' % (len(rel_valid)))
    print('total relation-test: %d ...' % (len(rel_test)))
    prepare.save_corpus(dstdir + 'corpus.txt', corpus)

    prepare.save_relation(dstdir + 'relation_train.txt', rel_train)
    prepare.save_relation(dstdir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(dstdir + 'relation_test.txt', rel_test)
    print('Preparation finished ...')

    preprocessor = Preprocess(word_stem_config={'enable': False}, word_filter_config={'min_freq': 2})
    dids, docs, posids, phraseids = preprocessor.run_orig(dstdir + 'corpus.txt')
    preprocessor.save_pos_dict(dstdir + 'pos_dict.txt', True)
    preprocessor.save_phrase_dict(dstdir + 'phrase_dict.txt', True)

    # preprocessor.save_pos_stats(dstdir + 'pos_stats.txt', True)
    preprocessor.save_word_dict(dstdir + 'word_dict.txt', True)
    preprocessor.save_words_stats(dstdir + 'word_stats.txt', True)

    fout = open(dstdir + 'corpus_preprocessed.txt', 'w')
    fout_pos = open(dstdir + 'corpus_postag_preprocessed.txt','w')
    fout_phrase = open(dstdir + 'corpus_phrase_preprocessed.txt','w')
    for inum, did in enumerate(dids):
        try:
            id_list = list(map(str,docs[inum]))
            pos_list = list(map(str,posids[inum]))
            phrase_list = list(map(str,phraseids[inum]))
            fout.write('%s %s %s\n' % (did, len(id_list), ' '.join(id_list)))
            fout_pos.write('%s %s %s\n' % (did, len(pos_list), ' '.join(pos_list)))
            fout_phrase.write('%s %s %s\n' % (did, len(phrase_list), ' '.join(phrase_list)))
        except Exception:
            pass
    fout.close()
    fout_pos.close()
    print('Preprocess finished ...')

    #dssm_corp_input = dstdir + 'corpus_preprocessed.txt'
    #dssm_corp_output = dstdir + 'corpus_preprocessed_dssm.txt'
    word_dict_input = dstdir + 'word_dict.txt'
    triletter_dict_output = dstdir + 'triletter_dict.txt'
    word_triletter_output = dstdir + 'word_triletter_map.txt'
    word_dict = read_dict(word_dict_input)
    word_triletter_map = {}
    triletter_stats = {}
    for wid, word in word_dict.items():
        nword = '#' + word + '#'
        ngrams = NgramUtil.ngrams(list(nword), 3, '')
        word_triletter_map[wid] = []
        for tric in ngrams:
            if tric not in triletter_stats:
                triletter_stats[tric] = 0
            triletter_stats[tric] += 1
            word_triletter_map[wid].append(tric)
    triletter_dict = filter_triletter(triletter_stats, 5, 10000)
    with open(triletter_dict_output, 'w') as f:
        for tri_id, tric in triletter_dict.items():
            print >> f, tri_id, tric
    with open(word_triletter_output, 'w') as f:
        for wid, trics in word_triletter_map.items():
            print >> f, wid, ' '.join([str(triletter_dict[k]) for k in trics if k in triletter_dict])

    print('Triletter Processing finished ...')

