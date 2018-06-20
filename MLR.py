import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer

from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec

from keras.preprocessing.sequence import pad_sequences

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

import json
import nltk,string

import os

from sklearn import svm

MAX_NB_WORDS=99
MAX_DOC_LEN=200
EMBEDDING_DIM=200
DOCVECTOR_MODEL="docvector_model"

class MLR(object):
    
    def __init__(self, data): 
        self.data = data;
    
    def pretrain(self, RETRAIN=0):
        with open("word_sample.json", 'r') as f:
            reviews=[]
            for line in f: 
                review = json.loads(line) 
                try:
                    review["text"].strip().lower().encode('ascII')
                except:
                    # do nothing
                    a = 1
                else:
                    reviews.append(review["text"])

        sentences=[ [token.strip(string.punctuation).strip() \
                     for token in nltk.word_tokenize(doc.lower()) \
                         if token not in string.punctuation and \
                         len(token.strip(string.punctuation).strip())>=2]\
                     for doc in reviews]

        docs=[TaggedDocument(sentences[i], [str(i)]) for i in range(len(sentences)) ]
        
        if RETRAIN==0 and os.path.exists(DOCVECTOR_MODEL):
            self.wv_model = doc2vec.Doc2Vec.load(DOCVECTOR_MODEL)
#             print self.wv_model
        else:
            self.wv_model = doc2vec.Doc2Vec(dm=1, min_count=5, window=5, size=200, workers=4)
            self.wv_model.build_vocab(docs)
            for epoch in range(30):
                # shuffle the documents in each epoch
                shuffle(docs)
                # in each epoch, all samples are used
                self.wv_model.train(docs, total_examples=len(docs), epochs=1)
                
            self.wv_model.save(DOCVECTOR_MODEL)

#         print("Top 5 words similar to word 'price'")
#         print self.wv_model.wv.most_similar('price', topn=5)

#         print("Top 5 words similar to word 'price' but not relevant to 'bathroom'")
#         print self.wv_model.wv.most_similar(positive=['price','money'], negative=['bathroom'], topn=5)

#         print("Similarity between 'price' and 'bathroom':")
#         print self.wv_model.wv.similarity('price','bathroom') 

#         print("Similarity between 'price' and 'charge':")
#         print self.wv_model.wv.similarity('price','charge') 

#         print self.wv_model.wv

    def checkLabelsPerform(self):
        labels = []
        # fetch labels for each sentence        
        for subdata in self.data[2][0:500]:
            label = []
            for d in subdata.split(","):
                label.append(float(d.strip()))
            labels.append(label)

        Y = np.copy(labels)

        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        NUM_WORDS = min(MAX_NB_WORDS, len(tokenizer.word_index))
        embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))

        for word, i in tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in self.wv_model.wv:
                embedding_matrix[i]=self.wv_model.wv[word]

        voc=tokenizer.word_index
        
        sequences = tokenizer.texts_to_sequences(self.data[1][0:500])
        padded_sequences = pad_sequences(sequences, \
                                         maxlen=MAX_DOC_LEN, \
                                         padding='post', truncating='post')
        self.label_padding_sequence = padded_sequences
        
        X_train, X_test, Y_train, Y_test = train_test_split(\
                        padded_sequences[0:500], Y[0:500], test_size=0.3, random_state=0)
        
        clf = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        clf.fit(X_train, Y_train)
        
        predicted = clf.predict(X_test)
        predicted = np.round(predicted, decimals=1)
        Y_actual = Y_test
        return mean_squared_error(Y_actual, predicted)

    def checkSentimentPerform(self):
        labels = []
        for i,subdata in enumerate(self.data[3][0:500]):
            labels.append([subdata])

        Y_labels = np.copy(labels)
        Y = Y_labels
        
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        NUM_WORDS = min(MAX_NB_WORDS, len(tokenizer.word_index))
        embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))

        for word, i in tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in self.wv_model.wv:
                embedding_matrix[i]=self.wv_model.wv[word]

        voc=tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(self.data[1][0:500])
        padded_sequences = pad_sequences(sequences, \
                                         maxlen=MAX_DOC_LEN, \
                                         padding='post', truncating='post')
        self.sent_padding_sequence = padded_sequences

        X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences[0:500], Y[0:500], test_size=0.3, random_state=0)
        
        clf = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
        clf.fit(X_train, Y_train)
        
        predicted = clf.predict(X_test)
        predicted = np.round(predicted, decimals=1)
        Y_actual = Y_test
        return mean_squared_error(Y_actual, predicted)
