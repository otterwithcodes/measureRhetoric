#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:04:37 2022


1. appeal to authority/endorsement
2. appeal to history
3. appeal to national greatness
4. cost/benifit
5. morality
6. public opinion

@author: liuyangdong
"""

import pandas as pd

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    
    classifier.fit(feature_vector_train, label)
    
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        
        predictions = predictions.argmax(axis=-1)
        
    return metrics.accuracy_score(predictions, valid_y)
    



policy = pd.read_csv("dataFile_label.csv")

dataset = policy.loc[:,['text_one', 'element_one']]

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(dataset['text_one'], dataset['element_one'])



# label编码为目标变量
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


"""
rhetoric = ['appeal to authority/endorsement', 'appeal to history', 'appeal to national greatness', 
            'cost/benifit', 'morality', 'public opinion']
"""

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)

tfidf_vect.fit(dataset['text_one'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)


tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(dataset['text_one'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)


#特征为词语的IT-IDF的NaiveBayes分类器
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NaiveBayes, WordLevel TF-IDF: ", accuracy)

#特征为N-Gram的IT-IDF的NaiveBayes分类器
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NaiveBayes, N-Gram Vectors: ", accuracy)


#特征为N-Gram的IT-IDF的SVM分类器
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)


#特征为词语级别TF-IDF向量的线性分类器
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LogisticRegression, WordLevel TF-IDF: ", accuracy)


#特征为多个词语级别TF-IDF向量的线性分类器
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LogisticRegression, N-Gram Vectors: ", accuracy)







