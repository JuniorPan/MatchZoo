# -*- coding: utf-8 -*-

__author__ = 'haolanchen'

from nltk import word_tokenize, pos_tag, ne_chunk, Tree, SnowballStemmer
from sys import argv

stemmer = SnowballStemmer('english')
feature_word_set = {w for w in 'in,as,by,for,on,from,which,at,what,also,who,why,many,can,after,when,where,during,than,time,how,since,including,years'.split(',')}

def sent_ner(s):
    return ne_chunk(pos_tag(word_tokenize(s)))

def tokens_ner(token_list):
    return ne_chunk(pos_tag([t.encode('utf8') for t in token_list]))

def get_ner_postag_list(t):
    if not isinstance(t,Tree):
        return [t[1]] # POS TAG of POS TAGGING tuple
    else:
        label = t.label()
        postag_list = list(reduce(lambda x,y: x+y, [[]]+[get_ner_postag_list(subt) for subt in t]))
        labeled_postag_list = ['{}_{}'.format(label, postag) for postag in postag_list]
        return labeled_postag_list

def get_featured_ner_postag_list(t):
    if not isinstance(t,Tree):
        word = t[0].lower()
        tag = t[1]
        res = '{}_{}'.format(tag,word) if word in feature_word_set else tag
        return [res] # POS TAG of POS TAGGING tuple
    else:
        label = t.label()
        postag_list = list(reduce(lambda x,y: x+y, [[]]+[get_featured_ner_postag_list(subt) for subt in t]))
        labeled_postag_list = ['{}_{}'.format(label, postag) for postag in postag_list]
        return labeled_postag_list

if '__main__' == __name__:
    s = argv[1]
    t = sent_ner(s)
    postag_list = get_featured_ner_postag_list(t)
    print(t)
    print(postag_list)
    print(feature_word_set)
