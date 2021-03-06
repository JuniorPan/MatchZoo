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

import sys
sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from DynamicMaxPooling import *
from utility import *


class Geminet(BasicModel):
    def __init__(self, config):
        super(Geminet, self).__init__(config)
        self.__name = 'Geminet'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[Geminet] parameter check wrong')
        print('[Geminet] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 32)
        self.set_default('kernel_size', [3, 3])
        self.set_default('dpool_size', [3, 10])
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        query_pos = Input(name='query_pos', shape=(self.config['pos1_maxlen'],))
        show_layer_info('Input', query_pos)

        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)
        doc_pos = Input(name='doc_pos', shape=(self.config['pos2_maxlen'],))
        show_layer_info('Input', doc_pos)

        dpool_index = Input(name='dpool_index', shape=[self.config['text1_maxlen'], self.config['text2_maxlen'], 3], dtype='int32')
        show_layer_info('Input', dpool_index)

        dpool_pos_index = Input(name='dpool_pos_index', shape=[self.config['pos1_maxlen'], self.config['pos2_maxlen'], 3],
                            dtype='int32')
        show_layer_info('Input', dpool_pos_index)

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable = self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        pos_embedding = Embedding(self.config['pos_vocab_size'], self.config['pos_embed_size'], weights=[self.config['pos_embed']],
                              trainable=self.embed_trainable)
        q_pos_embed = pos_embedding(query_pos)
        show_layer_info('Embedding', q_pos_embed)
        d_pos_embed = pos_embedding(doc_pos)
        show_layer_info('Embedding', d_pos_embed)

        # q_merged = Concatenate()([q_embed, q_pos_embed])
        # d_merged = Concatenate()([d_embed, d_pos_embed])

        '''
            CNN for word matrix
        '''
        cross = Dot(axes=[2, 2], normalize=False)([q_embed, d_embed])
        show_layer_info('Dot', cross)
        cross_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(cross)
        show_layer_info('Reshape', cross_reshape)

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
            CNN for POS matrix
        '''
        cross_pos = Dot(axes=[2, 2], normalize=False)([q_pos_embed, d_pos_embed])
        show_layer_info('Dot_pos', cross_pos)
        cross_reshape_pos = Reshape((self.config['pos1_maxlen'], self.config['pos2_maxlen'], 1))(cross_pos)
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

        cross_pos_k = Lambda(lambda x: K.tf.nn.top_k(x, 5, sorted=True)[0])(cross_pos) # fixme
        cross_pos_k = Flatten()(cross_pos_k)
        cross_pos_k =  Dropout(rate=self.config['dropout_rate'])(cross_pos_k)

        '''
            Concat two representations
        '''
        concat = Concatenate()([pool1_flat_drop, cross_pos_k]) # fixme
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(concat)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(concat)
        show_layer_info('Dense', out_)

        model = Model(inputs=[query, query_pos, doc, doc_pos, dpool_index, dpool_pos_index], outputs=out_)
        return model
