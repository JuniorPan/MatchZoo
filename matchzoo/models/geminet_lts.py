# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers import Reshape, Embedding, Merge, Dot
from keras.activations import softmax
from keras.optimizers import Adam
from model import BasicModel

import sys

sys.path.append('../matchzoo/layers/')
sys.path.append('../matchzoo/utils/')
from DynamicMaxPooling import *
from utility import *



class Gemini(BasicModel):
    def __init__(self, config):
        super(Gemini, self).__init__(config)
        self.__name = 'Gemini'
        self.check_list = [ 'text1_maxlen', 'text2_maxlen',
                            'pos1_maxlen', 'pos2_maxlen',
                   'embed', 'embed_size', 'vocab_size',
                   'kernel_size', 'kernel_count',
                   'dpool_size', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[Gemini] parameter check wrong')
        print('[Gemini] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('kernel_count', 64)
        self.set_default('kernel_size', [3, 3])
        self.set_default('dpool_size', [3, 10])
        self.set_default('dropout_rate', 0)
        self.set_default('topk', 20)
        self.config.update(config)

    def build(self):
        # vocab input
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        # POS input
        query_pos = Input(name='query_pos', shape=(self.config['pos1_maxlen'],))
        show_layer_info('Input', query_pos)
        doc_pos = Input(name='doc_pos', shape=(self.config['pos2_maxlen'],))
        show_layer_info('Input', doc_pos)

        # vocab embedding
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.embed_trainable)
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)

        # POS embedding
        pos_embedding = Embedding(self.config['pos_vocab_size'], self.config['pos_embed_size'],
                                  weights=[self.config['pos_embed']],
                                  trainable=self.embed_trainable)
        q_pos_embed = pos_embedding(query_pos)
        show_layer_info('Embedding', q_pos_embed)
        d_pos_embed = pos_embedding(doc_pos)
        show_layer_info('Embedding', d_pos_embed)


        # compute vocab gating
        w_g = Dense(1)(q_embed)
        show_layer_info('Dense', w_g)
        g = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['text1_maxlen'],))(w_g)
        show_layer_info('Lambda-softmax', g)
        g = Reshape((self.config['text1_maxlen'],))(g)
        show_layer_info('Reshape', g)

        # vocab networking
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        show_layer_info('Dot', mm)
        mm_k = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(mm)
        show_layer_info('Lambda-topk', mm_k)

        for i in range(self.config['num_layers']):
            mm_k = Dense(self.config['hidden_sizes'][i], activation='softplus', kernel_initializer='he_uniform',
                         bias_initializer='zeros')(mm_k)
            show_layer_info('Dense', mm_k)

        mm_k_dropout = Dropout(rate=self.config['dropout_rate'])(mm_k)
        show_layer_info('Dropout', mm_k_dropout)

        mm_reshape = Reshape((self.config['text1_maxlen'],))(mm_k_dropout)
        show_layer_info('Reshape', mm_reshape)

        mean = Dot(axes=[1, 1])([mm_reshape, g])
        show_layer_info('Dot', mean)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(mean)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Reshape((1,))(mean)
        show_layer_info('Dense', out_)

        # compute POS gating
        # w_g_pos = Dense(1)(q_pos_embed)
        # show_layer_info('Dense', w_g_pos)
        # g_pos = Lambda(lambda x: softmax(x, axis=1), output_shape=(self.config['pos1_maxlen'],))(w_g_pos)
        # show_layer_info('Lambda-softmax', g_pos)
        # g_pos = Reshape((self.config['pos1_maxlen'],))(g_pos)
        # show_layer_info('Reshape', g_pos)

        # POS networking
        mm_pos = Dot(axes=[2, 2], normalize=True)([q_pos_embed, d_pos_embed])
        show_layer_info('Dot', mm_pos)
        mm_k_pos = Lambda(lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[0])(mm_pos)
        show_layer_info('Lambda-topk', mm_k_pos)

        for i in range(self.config['num_layers']):
            mm_k_pos = Dense(self.config['hidden_sizes'][i], activation='softplus', kernel_initializer='he_uniform',
                         bias_initializer='zeros')(mm_k_pos)
            show_layer_info('Dense', mm_k_pos)

        mm_k_pos_dropout = Dropout(rate=self.config['dropout_rate'])(mm_k_pos)
        show_layer_info('Dropout', mm_k_pos_dropout)

        mm_pos_reshape = Reshape((self.config['pos1_maxlen'],))(mm_k_pos_dropout)
        show_layer_info('Reshape', mm_pos_reshape)

        # mean_pos = Dot(axes=[1, 1])([mm_pos_reshape, g_pos])
        # show_layer_info('Dot', mean_pos)

        concat = Concatenate()([mean, mm_pos_reshape])
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(mean)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Reshape((1,))(mean)
        show_layer_info('Dense', out_)


        model = Model(inputs=[query, doc, query_pos, doc_pos], outputs=out_)
        return model
