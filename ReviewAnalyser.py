from string import punctuation
from gensim.models.doc2vec import TaggedDocument
import json
import gensim
import nltk,string
from random import shuffle
from gensim.models import doc2vec
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, \
Dropout, Activation, Input, Flatten, Concatenate
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk import tokenize
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import os
import matplotlib  
matplotlib.use('Agg') 
from matplotlib.pyplot import plot,savefig 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from keras.models import load_model

DOCVECTOR_MODEL="docvector_model"
BEST_MODEL_FILEPATH="best_model"
BEST_LABEL_WEIGHT_FILEPATH="best_label_weight"
BEST_SENT_MODEL_FILEPATH="best_sent_model"
BEST_SENT_WEIGHT_FILEPATH="best_sent_weight"
QUALITY_MODEL="quality_model"
MAX_NB_WORDS=99
MAX_DOC_LEN=200
EMBEDDING_DIM=200
FILTER_SIZES=[2,3,4]
BTACH_SIZE = 64
NUM_EPOCHES = 40
LABELS = ['amenities','environment','food','location','price','service','sentiment']

class ReviewAnalyser(object):
    
    # review's ann model
    ann_model = None
    # label's cnn model
    label_model = None
    # labels input padding sequence
    label_padding_sequence = None
    # labels actual classification
    label_act = None
    # labels test set feature
    label_X_test = None
    # labels test set labels
    label_Y_set = None
    # labels validation set feature
    label_X_train = None
    # labels validation set labels
    label_Y_train = None
    # sentiment's cnn model
    sent_model = None
    # sentiment input padding sequence
    sent_padding_sequence = None
    # sentiment actual classification
    sent_act = None
    # labels test set
    sent_test_set = None
    # sentiment's validation set
    sent_validation_set = None
    # sentitment's test set feature
    sent_X_test = None
    # sentitment's test set labels
    sent_Y_set = None
    # sentitment's validation set feature
    sent_X_train = None
    # sentitment's validation set labels
    sent_Y_train = None
    # doc2vector's cnn model
    wv_model = None
    
    def __init__(self, data): 
        self.data = data;
        
    @staticmethod    
    def cnn_model(FILTER_SIZES, \
        # filter sizes as a list
        MAX_NB_WORDS, \
        # total number of words
        MAX_DOC_LEN, \
        # max words in a doc
        NUM_OUTPUT_UNITS=1, \
        # number of output units
        EMBEDDING_DIM=200, \
        # word vector dimension
        NUM_FILTERS=64, \
        # number of filters for all size
        DROP_OUT=0.5, \
        # dropout rate
        PRETRAINED_WORD_VECTOR=None,\
        # Whether to use pretrained word vectors
        LAM=0.01,\
        ACTIVATION='sigmoid'):            
        # regularization coefficient
    
        main_input = Input(shape=(MAX_DOC_LEN,), \
                           dtype='int32', name='main_input')

        if PRETRAINED_WORD_VECTOR is not None:
            embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                            output_dim=EMBEDDING_DIM, \
                            input_length=MAX_DOC_LEN, \
                            weights=[PRETRAINED_WORD_VECTOR],\
                            trainable=False,\
                            name='embedding')(main_input)
        else:
            embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                            output_dim=EMBEDDING_DIM, \
                            input_length=MAX_DOC_LEN, \
                            name='embedding')(main_input)

        conv_blocks = []
        for f in FILTER_SIZES:
            conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                          activation='relu', name='conv_'+str(f))(embed_1)
            conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
            conv = Flatten(name='flat_'+str(f))(conv)
            conv_blocks.append(conv)

        z=Concatenate(name='concate')(conv_blocks)
        drop=Dropout(rate=DROP_OUT, name='dropout')(z)

        dense = Dense(192, activation='relu',\
                        kernel_regularizer=l2(LAM),name='dense')(drop)
        preds = Dense(NUM_OUTPUT_UNITS, activation=ACTIVATION, name='output')(dense)
        model = Model(inputs=main_input, outputs=preds)

#         model.compile(loss="binary_crossentropy", \
#                   optimizer="adam", metrics=["accuracy"])
        
        model.compile(loss="mean_squared_error", \
                  optimizer="adam", metrics=["accuracy"]) 


        return model

    # training to change document into vector using gensim
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

    # training labels CNN
    def trainLebels(self, RETRAIN=0):
        labels = []
        # fetch labels for each sentence        
        for subdata in self.data[2][0:500]:
            label = []
            for d in subdata.split(","):
                label.append(float(d.strip()))
            labels.append(label)
        
        Y = np.copy(labels)
        self.label_act = Y

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
        
        self.label_X_train = X_train
        self.label_Y_train = Y_train
        self.label_X_test = X_test
        self.label_Y_test = Y_test
        
        if(RETRAIN == 0 and os.path.exists(BEST_MODEL_FILEPATH)):
#                 self.label_model.load_weights(BEST_MODEL_FILEPATH)
                self.label_model = load_model(BEST_MODEL_FILEPATH)
#                 pred=self.label_model.predict(padded_sequences[0:500])
                return
        
        self.label_model=ReviewAnalyser.cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
                        MAX_DOC_LEN, NUM_OUTPUT_UNITS=6, \
                        PRETRAINED_WORD_VECTOR=embedding_matrix)

        earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
        checkpoint = ModelCheckpoint(BEST_LABEL_WEIGHT_FILEPATH, monitor='val_acc', \
                                     verbose=2, save_best_only=True, mode='max')
        
        training=self.label_model.fit(X_train, Y_train, \
                  batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
                  callbacks=[earlyStopping, checkpoint],\
                  validation_data=[X_test, Y_test], verbose=2)
        
        self.label_model.save(BEST_MODEL_FILEPATH)
        
        return
        
    # training sentiment CNN        
    def trainSentiment(self, RETRAIN=0):
        labels = []
        for i,subdata in enumerate(self.data[3][0:500]):
            labels.append([subdata])

        Y_labels = np.copy(labels)
        Y = Y_labels
        self.sent_act = Y
        
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
        self.sent_X_train = X_train
        self.sent_X_test = X_test
        self.sent_Y_train = Y_train
        self.sent_Y_test = Y_test
        
        if(RETRAIN == 0 and os.path.exists(BEST_SENT_MODEL_FILEPATH)):
                self.sent_model = load_model(BEST_SENT_MODEL_FILEPATH)
                pred=self.sent_model.predict(padded_sequences[0:500])
                return
        
        
        self.sent_model=ReviewAnalyser.cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
                    MAX_DOC_LEN, \
                    PRETRAINED_WORD_VECTOR=embedding_matrix)

        earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
        checkpoint = ModelCheckpoint(BEST_SENT_WEIGHT_FILEPATH, monitor='val_acc', \
                                     verbose=2, save_best_only=True, mode='max')

        training=self.sent_model.fit(X_train, Y_train, \
                  batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
                  callbacks=[earlyStopping, checkpoint],\
                  validation_data=[X_test, Y_test], verbose=2) 
        
        self.sent_model.save(BEST_SENT_MODEL_FILEPATH)
        
        return
    
    def checkLabelPerform(self):
        predicted=self.label_model.predict(self.label_X_test)
        predicted = np.round(predicted, decimals=1)
        Y_actual = self.label_Y_test
        return mean_squared_error(Y_actual, predicted)
        

    def checkSentimentPerform(self):
        pred=self.sent_model.predict(self.sent_X_test)
        predicted=np.reshape(pred, -1)
        predicted = np.round(predicted, decimals=1)
        Y_actual = self.sent_Y_test
        return mean_squared_error(Y_actual, predicted)
        
       
    # check document information to determine the value of hyper-parameter
    def checkDocInform(self):  
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        total_nb_words=len(tokenizer.word_counts)
        sequences = tokenizer.texts_to_sequences(self.data[1][0:500])
        print "\n############## document information ##############\n"
        print "total_nb_words:"
        print(total_nb_words)

        word_counts=pd.DataFrame(\
                    tokenizer.word_counts.items(), \
                    columns=['word','count'])
        df=word_counts['count'].value_counts().reset_index()
        df['percent']=df['count']/len(tokenizer.word_counts)
        df['cumsum']=df['percent'].cumsum()

        plt.bar(df["index"].iloc[0:50], df["percent"].iloc[0:50])
        plt.plot(df["index"].iloc[0:50], df['cumsum'].iloc[0:50], c='green')

        plt.xlabel('Word Frequency')
        plt.ylabel('Percentage')
        savefig('word_freq.jpg')
        plt.show()
        plt.close('all')
        
        sen_len=pd.Series([len(item) for item in sequences])

        df=sen_len.value_counts().reset_index().sort_values(by='index')
        df.columns=['index','counts']

        df=df.sort_values(by='index')
        df['percent']=df['counts']/len(sen_len)
        df['cumsum']=df['percent'].cumsum()
        
        plt.plot(df["index"], df['cumsum'], c='green')

        plt.xlabel('Sentence Length')
        plt.ylabel('Percentage')
        savefig('sent_len.jpg')
        plt.show()
        plt.close('all')
        return
        
    # predict labels for text, need to execute trainLabels first
    def predictLabels(self, text_arr=[]):
        if len(text_arr)==0:
            return
        rtn = {}
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        sub_sequences = tokenizer.texts_to_sequences(text_arr)
        padded_sub_sequences = pad_sequences(sub_sequences, \
                                 maxlen=MAX_DOC_LEN, \
                                 padding='post', truncating='post')
        sub_pred = self.label_model.predict(padded_sub_sequences)
        for i, key in enumerate(text_arr):
            dict1 = {}
            pred_list = sub_pred[i].tolist()
            for i, sub_pred_list in enumerate(pred_list):
                dict1[LABELS[i]] = pred_list[i]
            rtn[key] = dict1
        return rtn
        
    # predict sentiments for text, need to execute trainSentiment first    
    def predictSentiment(self, text_arr=[]):
        if len(text_arr)==0:
            return
        rtn = {}
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.data[1][0:500])
        sub_sequences = tokenizer.texts_to_sequences(text_arr)
        padded_sub_sequences = pad_sequences(sub_sequences, \
                                 maxlen=MAX_DOC_LEN, \
                                 padding='post', truncating='post')
        sub_pred = self.sent_model.predict(padded_sub_sequences)
        for i, key in enumerate(text_arr):
            rtn[key] = sub_pred[i].tolist()[0]
        return rtn
    
    # predict quality for reviews, need to execute trainLabels,trainSentiment and trainQuality first    
    def predictQuality(self, review_arr=[]):
        text_arr=[]
        sentence_review_mapping = []
        data = []
        rows = {}
        if len(review_arr)==0:
            return
        for i, rev in enumerate(review_arr):
            rev_sent = tokenize.sent_tokenize(rev)
            for sent in rev_sent:
                text_arr.append(sent)
                sentence_review_mapping.append((i,sent))
            
        label_predict = self.predictLabels(text_arr)
        sentiment_predict = self.predictSentiment(text_arr)
    
        for mapping in sentence_review_mapping:
            rows[mapping[1]] = {}
            rows[mapping[1]]["review_id"] = mapping[0]
            rows[mapping[1]]["sentence"] = mapping[1]
            tmp = label_predict[mapping[1]]
            rows[mapping[1]]["labels"] = str(tmp["amenities"])+','+str(tmp["environment"])+','+str(tmp["food"])+','+str(tmp["location"])+','+str(tmp["price"])+','+str(tmp["service"])
            rows[mapping[1]]["sentiment"] = sentiment_predict[mapping[1]]
        
        data = []
        for key in rows:
            subdata=[]
            subdata.append(rows[key]["review_id"])
            subdata.append(rows[key]["sentence"])
            subdata.append(rows[key]["labels"])
            subdata.append(rows[key]["sentiment"])
            data.append(subdata)
        df=pd.DataFrame(data, columns=["review_id","sentence","labels","sentiment"])
        res = self.gradeReview(df.values.tolist())
        
        predicted = {}
        for k in res:
            res[k]['quality'] = res[k]['quality']/res[k]['items']
            predicted[review_arr[k]] = res[k]['quality']
        
        rtn = {
            "label_predict": label_predict,
            "sentiment_predict": sentiment_predict,
            "review_predict": predicted
        }
        return rtn

    # for analysing the data sample
    def dataSamplePlt(self):
        x = np.arange(0, 500, 1);
        y = np.copy(self.data[3][0:500])
        plt.xlabel('data sample items')
        plt.ylabel('sentiment')
        plt.plot(x, y,'ro',label="the level of objectivity")
        plt.legend(loc='lower right')
        savefig('data_sample_sentiment.jpg')
        plt.close('all')
        
        amenities = []
        environment = []
        food = []
        location = []
        price = []
        service = []
        for subdata in self.data[2][0:500]:
            label = []
            for key,value in enumerate(subdata.split(",")):
                if key == 0:
                    amenities.append(float(value.strip()))
                if key == 1:
                    environment.append(float(value.strip()))
                if key == 2:
                    food.append(float(value.strip()))
                if key == 3:
                    location.append(float(value.strip()))
                if key == 4:
                    price.append(float(value.strip()))
                if key == 5:
                    service.append(float(value.strip()))
        plt.xlabel('data sample items')
        plt.ylabel('labels')
        plt.scatter(x,np.copy(amenities),color='red',label="amenities")
        plt.scatter(x,np.copy(environment),color='green',label="environment")
        plt.scatter(x,np.copy(food),color='blue',label="food")
        plt.scatter(x,np.copy(location),color='yellow',label="location")
        plt.scatter(x,np.copy(price),color='black',label="price")
        plt.scatter(x,np.copy(service),color='orange',label="service")
        plt.legend(loc='lower right')
        savefig('data_sample_labels.jpg')
        plt.close('all')
        return 
    
    def gradeCSV(self):
        rows = self.gradeReview(self.data[0:500].values.tolist())
        for k in rows:
            rows[k]['quality'] = rows[k]['quality']/rows[k]['items']
            self.data.loc[self.data[0] == k, 4] = rows[k]['quality']
        print self.data.head(10)
        self.data.to_csv('data_sample3.csv')
    
    
    def gradeReview(self, reviews):
        rows = {}
        for subdata in reviews:
            sent = subdata[3]
            labels = subdata[2].split(',')
            if rows.has_key(subdata[0]):
                rows[subdata[0]]["items"] = float(rows[subdata[0]]["items"]) + 1
                for key,label in enumerate(labels):
                    rows[subdata[0]]['quality'] = rows[subdata[0]]['quality'] + (float(label)*float(sent))
            else:
                rows[subdata[0]] = {
                    'items': 1.0,
                    'quality': 0.0
                }
                for key,label in enumerate(labels):
                    rows[subdata[0]]['quality'] = rows[subdata[0]]['quality'] + (float(label)*float(sent))
                
        return rows
