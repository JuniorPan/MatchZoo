#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'haolanchen'


import logging
import sys


logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s')

log = logging.getLogger('')


class Logger(object):

    def __init__(self):
        log.info('Initializing {}'.format(self.__class__.__name__))

