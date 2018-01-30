# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Merge, Dot
from keras.optimizers import Adam
from model import BasicModel
from layers.DynamicMaxPooling import *
from utils.utility import *


class MatchPyramid(BasicModel):
    def __init__(self, config):
        super(MatchPyramid, self).__init__(config)
        self.__name = 'MatchPyramid'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MatchPyramid] parameter check wrong')
        print('[MatchPyramid] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', [3, 3])
        self.set_default('dpool_size', [3, 10])
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):

        '''
        Constructing dpool channel
        '''
        dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3],
                            dtype='int32')
        show_layer_info('Input', dpool_index)

        dpool_pos_index = Input(name='dpool_pos_index',
                                shape=[self.config['pos1_maxlen'], self.config['pos2_maxlen'], 3],
                                dtype='int32')
        show_layer_info('Input', dpool_pos_index)

        dpool_phrase_index = Input(name='dpool_phrase_index',
                                shape=[self.config['phrase1_maxlen'], self.config['phrase2_maxlen'], 3],
                                dtype='int32')
        show_layer_info('Input', dpool_phrase_index)

        '''
        Constructing term channel
        '''
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        cross = Dot(axes=[2, 2], normalize=False)([q_embed, d_embed])

        '''
        Constructing attention channel based on term weight
        '''
        idf_embedding = Embedding(self.config['vocab_size'], 1, weights=[self.config['idf']], trainable = self.embed_trainable)
        q_idf = idf_embedding(query)
        show_layer_info('query idf', q_idf)
        d_idf = idf_embedding(doc)
        show_layer_info('doc idf', d_idf)

        tw_att = Dot(axes=[2, 2], normalize=True)([q_idf, d_idf])

        '''
        join term channel with term weight attention channel
        '''
        tw_term_cross = Multiply()([cross, tw_att])

        show_layer_info('Dot', tw_term_cross)
        tw_term_cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(tw_term_cross)
        show_layer_info('Reshape', tw_term_cross_reshape)

        conv2d = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])

        conv1 = conv2d(cross_reshape)
        show_layer_info('Conv2D', conv1)
        pool1 = dpool([conv1, dpool_index])
        show_layer_info('DynamicMaxPooling', pool1)
        pool1_flat = Flatten()(pool1)
        show_layer_info('Flatten', pool1_flat)
        pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(pool1_flat)
        show_layer_info('Dropout', pool1_flat_drop)


        '''
        Constructing attention channel based on postag-ner
        '''
        query_pos = Input(name='query_pos', shape=(self.config['pos1_maxlen'],))
        show_layer_info('Input', query_pos)

        doc_pos = Input(name='doc_pos', shape=(self.config['pos2_maxlen'],))
        show_layer_info('Input', doc_pos)

        pos_embedding = Embedding(self.config['pos_vocab_size'], self.config['pos_embed_size'],
                                  weights=[self.config['pos_embed']],
                                  trainable=self.embed_trainable)
        q_pos_embed = pos_embedding(query_pos)
        show_layer_info('Embedding', q_pos_embed)
        d_pos_embed = pos_embedding(doc_pos)
        show_layer_info('Embedding', d_pos_embed)

        pos_att = Dot(axes=[2, 2], normalize=False)([q_pos_embed, d_pos_embed])
        show_layer_info('Dot_pos', pos_att)


        '''
        join term channel with postag-ner attention channel
        '''
        pos_term_cross = Multiply()([cross, pos_att])

        cross_reshape_pos = Reshape((self.config['pos1_maxlen'], self.config['pos2_maxlen'], 1))(pos_term_cross)
        show_layer_info('Reshape_pos', cross_reshape_pos)

        conv2d_pos = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        dpool_pos = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])

        conv1_pos = conv2d_pos(cross_reshape_pos)
        show_layer_info('Conv2D_pos', conv1_pos)
        pool1_pos = dpool_pos([conv1_pos, dpool_pos_index])
        show_layer_info('DynamicMaxPooling_pos', pool1_pos)
        pool1_flat_pos = Flatten()(pool1_pos)
        show_layer_info('Flatten_pos', pool1_flat_pos)
        pool1_flat_drop_pos = Dropout(rate=self.config['dropout_rate'])(pool1_flat_pos)
        show_layer_info('Dropout_pos', pool1_flat_drop_pos)


        '''
        Constructing phrase channel
        '''
        query_phrase = Input(name='query_phrase', shape=(self.config['phrase1_maxlen'],))
        show_layer_info('Input', query_phrase)

        doc_phrase = Input(name='doc_phrase', shape=(self.config['phrase2_maxlen'],))
        show_layer_info('Input', doc_phrase)

        phrase_embedding = Embedding(self.config['phrase_vocab_size'], self.config['phrase_embed_size'],
                                  weights=[self.config['phrase_embed']],
                                  trainable=self.embed_trainable)
        q_phrase_embed = phrase_embedding(query_phrase)
        show_layer_info('Embedding', q_phrase_embed)
        d_phrase_embed = phrase_embedding(doc_phrase)
        show_layer_info('Embedding', d_phrase_embed)

        cross_phrase = Dot(axes=[2, 2], normalize=False)([q_phrase_embed, d_phrase_embed])
        show_layer_info('Dot_phrase', cross_phrase)
        cross_reshape_phrase = Reshape((self.config['phrase1_maxlen'], self.config['phrase2_maxlen'], 1))(cross_phrase)
        show_layer_info('Reshape_phrase', cross_reshape_phrase)

        conv2d_phrase = Conv2D(self.config['kernel_count'], self.config['kernel_size'], padding='same', activation='relu')
        dpool_phrase = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])

        conv1_phrase = conv2d_phrase(cross_reshape_phrase)
        show_layer_info('Conv2D_phrase', conv1_phrase)
        pool1_phrase = dpool_phrase([conv1_phrase, dpool_phrase_index])
        show_layer_info('DynamicMaxPooling_phrase', pool1_phrase)
        pool1_flat_phrase = Flatten()(pool1_phrase)
        show_layer_info('Flatten_phrase', pool1_flat_phrase)
        pool1_flat_drop_phrase = Dropout(rate=self.config['dropout_rate'])(pool1_flat_phrase)
        show_layer_info('Dropout_phrase', pool1_flat_drop_phrase)


        '''
        Concat phrase channel tw_att_term channel and pos_att_term channel
        '''
        concat = Concatenate()([pool1_flat_drop, pool1_flat_drop_pos, pool1_flat_drop_phrase])
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, query_pos, query_phrase, doc, doc_pos, doc_phrase, dpool_index, dpool_pos_index, dpool_phrase_index], outputs=out_)
        return model