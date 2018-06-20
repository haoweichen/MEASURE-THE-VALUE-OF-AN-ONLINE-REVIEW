#!flask/bin/python
from flask import Flask, jsonify, request, Response
from ReviewAnalyser import ReviewAnalyser
from MLR import MLR
from flask import render_template
import pandas as pd
from nltk import tokenize
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def test_api():
    dic1 = '{ "code": 100000, "predict": { "label_predict": { "the beef steak is good.": { "amenities": 0.00103157723788172, "environment": 0.0008686440996825695, "food": 0.9970982074737549, "location": 0.0010762664023786783, "null": 0.03090427815914154, "price": 0.007220499683171511, "service": 0.0008012578473426402 }, "the beer is good too.": { "amenities": 0.00982755422592163, "environment": 0.009218058548867702, "food": 0.5981521010398865, "location": 0.008505474776029587, "null": 0.20521660149097443, "price": 0.0184511486440897, "service": 0.002994242124259472 } }, "review_predict": [ 0.6158599853515625 ], "sentiment_predict": { "the beef steak is good.": 0.6330832242965698, "the beer is good too.": 0.822425127029419 } } }'
    predict = json.loads(dic1)
    html = render_template("predict.html", predict=predict)
    #resp = jsonify({'result': html})
    return html

@app.route('/reviewAnalyser/api/v1.0/predict/label', methods=['POST'])
def predict_label():
    data=pd.read_csv("data_sampl2.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainLebels(RETRAIN=0)
    label_predict = ra.predictLabels(text_arr=request.json.get("text_arr"))
    return jsonify({'code': 100000, 'data': label_predict}), 201

@app.route('/reviewAnalyser/api/v1.0/predict/sentiment', methods=['POST'])
def predict_sentiment():
    data=pd.read_csv("data_sample2.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainSentiment(RETRAIN=0)
    sentiment_predict = ra.predictSentiment(text_arr=request.json.get("text_arr"))
    return jsonify({'code': 100000, 'data': sentiment_predict}), 201

@app.route('/reviewAnalyser/api/v1.0/predict/review', methods=['POST'])
def predict_review():
    reviews = request.json.get("reviews")
    text_arr = []
    for rev in reviews[0:10]:
        rev_sent = tokenize.sent_tokenize(rev)
        for sent in rev_sent:
            text_arr.append(sent)

    data=pd.read_csv("data_sample2.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainLebels(RETRAIN=0)
    #label_predict = ra.predictLabels(text_arr)
    ra.trainSentiment(RETRAIN=0)
    #sentiment_predict = ra.predictSentiment(text_arr)
    prediction = ra.predictQuality(review_arr=reviews)
    html = render_template("predict.html", predict=prediction)
    #resp = jsonify({'code': 100000, 'labels' : label_predict,'sent': sentiment_predict})
    resp = jsonify({'code': 100000, 'predict' : prediction, 'html':html})
    return resp

@app.route('/reviewAnalyser/api/v1.0/performace/label', methods=['GET'])
def performance_labels():
    data=pd.read_csv("data_sample2.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainLebels(RETRAIN=0)
    rtn = ra.checkLabelPerform()
    return jsonify({'mean_squared_error': rtn})

@app.route('/reviewAnalyser/api/v1.0/performace/sent', methods=['GET'])
def performance_sent():
    data=pd.read_csv("data_sample2.csv",header=None)
    ra = ReviewAnalyser(data)
    ra.pretrain()
    ra.trainSentiment(RETRAIN=0)
    rtn = ra.checkSentimentPerform()
    return jsonify({'mean_squared_error': rtn})

@app.route('/reviewAnalyser/api/v1.0/mlr/performace/label', methods=['GET'])
def performance_labels_mlr():
    data=pd.read_csv("data_sample2.csv",header=None)
    mlr = MLR(data)
    mlr.pretrain()
    rtn = mlr.checkLabelsPerform()
    return jsonify({'mean_squared_error': rtn})

@app.route('/reviewAnalyser/api/v1.0/mlr/performace/sent', methods=['GET'])
def performance_sent_mlr():
    data=pd.read_csv("data_sample2.csv",header=None)
    mlr = MLR(data)
    mlr.pretrain()
    rtn = mlr.checkSentimentPerform()
    return jsonify({'mean_squared_error': rtn})

@app.route('/reviewAnalyser/api/v1.0/documentInform/1', methods=['GET'])
def performance_img1():
    image = file("word_freq.jpg")
    resp = Response(image, mimetype="image/jpeg")
    return resp

@app.route('/reviewAnalyser/api/v1.0/documentInform/2', methods=['GET'])
def performance_img2():
    image = file("sent_len.jpg")
    resp = Response(image, mimetype="image/jpeg")
    return resp

@app.route('/reviewAnalyser/api/v1.0/datasample/1', methods=['GET'])
def datasample_img1():
    image = file("data_sample_labels.jpg")
    resp = Response(image, mimetype="image/jpeg")
    return resp

@app.route('/reviewAnalyser/api/v1.0/datasample/2', methods=['GET'])
def datasample_img2():
    image = file("data_sample_sentiment.jpg")
    resp = Response(image, mimetype="image/jpeg")
    return resp

if __name__ == '__main__':

    app.run(debug=True, port=8887)


