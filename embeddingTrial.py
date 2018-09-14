# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:41:11 2017

@author: CTTC
"""

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
#from keras.layers.merge import concatenate
#from keras.models import Model
#from keras.layers.normalization import BatchNormalization
#from keras.callbacks import EarlyStopping, ModelCheckpoint

import sys
#rseload(sys)
#sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = 'embeddings/'
EMBEDDING_FILE = BASE_DIR + 'SBW-vectors-300-min5.bin.gz'

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=True)
