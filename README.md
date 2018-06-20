# ReviewAnalyser(using deep learning)

To choose the premium reviews with high quality. The details can be found in the presentation directory.

## deploying
download or check out the project, make sure the port 8887 and 3001 in the local are not in use.

## Usage

1. train the models first (follow the instruction below)
2. python ReviewAnalyser.py(this step can be skip if you dont want to train the model again)
3. python RestfulAPI.py(to test the restful api: use http://localhost:8887/)
4. cd admin; npm install; npm start;(access http://127.0.0.1:3001) account:admin pwd:admin

## how to train the models
open ReviewAnalyser.ipynb; go to the bottom of the file and set each RETRAIN = 1; run the file

## Entry point

./ReviewAnalyser.py
./RestfulAPI.py
./admin/app.js

## Input file

./data_sample2.csv (use to train the data)
./word_sample.json (which is the subset of yelp_data for the doc_vector training)

## Directory

 BIA660Final.postman_collection.json -- import this file to postman to test the restfulAPI
 BIA660_Final_present.pptx -- final presentation slides
 NBLabel.ipynb -- labels classification using Naive Bayes
 NBSentiment.ipynb -- sentiments classification using Naive Bayes
 README.md 
 RestfulAPI.ipynb  
 RestfulAPI.py -- RestfulAPI service: python RestfulAPI.py and access http://localhost:8887
 ReviewAnalyser.ipynb -- sentimens, labels and review quality classification using CNN and ANN
 ReviewAnalyser.py
 admin -- backend GUI: npm start and access http://localhost:3001 
 best_label_weight: -- label weight after training 
 best_model: -- label model after training
 best_sent_model 
 best_sent_weight
 data_sample2.csv -- sample data set
 docvector_model 
 midterm_presentation
 preprocessor.ipynb -- raw sample data preprocessor
 preprocessor.py
 quality_model
 templates -- backend templates
 word_sample.json -- data sample to train the document vector
 yelp_data_set -- yelp data set(too large to upload), download here https://www.yelp.com/dataset/download

## dependencies
python:
python 2.7
numpy
panda
gensim
nltk
sklearn
keras
matplot
etc..

nodejs:
express
uuid
request

## in the future
1. Add more trainning same (sample quantity)
2. Let more people to do the sample aggrement (sample quality) 
3. As long as the data sample incresing, the CNN training time processing increase as well. So we will put it into a distributed system
4. change the sentiment and review quality classification to regression. We can even change the labels classification to regression as well.




