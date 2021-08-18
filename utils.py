import numpy as np
from sklearn import (datasets, feature_extraction, linear_model, metrics)
from sklearn import neural_network
import re
import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
# from soylemma import Lemmatizer
from sklearn import svm
from sklearn import (tree, ensemble)
from sklearn.neural_network import MLPClassifier
from stopwords import stopword

def read_data():
    train = pd.read_csv('./data/train_data.csv')
    #     cate = pd.read_csv('./data/topic_dict.csv')
    X = np.array([x for x in train['title']])
    y = np.array([x for x in train['topic_idx']])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    return X_train, X_test, y_train, y_test


def preprocessing(X_train, stopwords):
    X_train['title'] = X_train['title'].str.replace("[^a-zA-Z가-힣 ]", " ")

    tokenizer = Okt()
    res = []
    for sentence in X_train['title']:
        tmp = []
        tmp = tokenizer.morphs(sentence)

        tokenized = []
        for token in tmp:
            if token not in stopwords and len(token) >= 2:
                tokenized.append(token)
        res.append(' '.join(tokenized))

    return res

def train(X_train, target, model):
    clf  = model
    clf.fit(X_train, target)

    return clf

def test(model, X_test, y_test):
    return model.score(X_test, y_test)

def vectorizor(dataset, dataset1):
    vectorizer = feature_extraction.text.TfidfVectorizer(sublinear_tf=True, smooth_idf=True, ngram_range=(1, 2))
    vectorizer.fit(dataset)
    x_data = vectorizer.transform(dataset)
    x_test = vectorizer.transform(dataset1)
    return x_data, x_test, vectorizer

def predict(model, x_test):
    pred = model.predict(x_test)
    return pred