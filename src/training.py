# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

#Import necessary dependencies and settings
import pandas as pd
import numpy as np
import logging
import re

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from cleaning import process_text

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
User defined functions to perform data preprocessing, feature engineering,
training the model  
"""

# Extracting Features
def extract_features(field,training_data,testing_data,type):
    """Extract features using different methods"""
    
    logging.info("Extracting features and creating vocabulary...")
    
    if "binary" in type:
        
        # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.95, analyzer=process_text)
        cv.fit_transform(training_data.values)
        
        train_feature_set=cv.transform(training_data.values)
        test_feature_set=cv.transform(testing_data.values)
        
        return train_feature_set,test_feature_set,cv
  
    elif "counts" in type:
        
        # COUNT BASED FEATURE REPRESENTATION
        cv= CountVectorizer(binary=False, max_df=0.95, analyzer=process_text)
        cv.fit_transform(training_data.values)
        train_feature_set=cv.transform(training_data.values)
        test_feature_set=cv.transform(testing_data.values)
        
        return train_feature_set,test_feature_set,cv
    else:    
        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95, analyzer=process_text)
        tfidf_vectorizer.fit_transform(training_data.values)
        
        train_feature_set=tfidf_vectorizer.transform(training_data.values)
        test_feature_set=tfidf_vectorizer.transform(testing_data.values)
        
        return train_feature_set,test_feature_set,tfidf_vectorizer


# Training model
def train_model(classifier, train_val, field="statement",feature_rep="binary"):
    """
    Training the classifier for the provided features.
    """
    logging.info("Starting model training...")   
    # GET A TRAIN TEST SPLIT (set seed for consistent results)
    training_data, testing_data = train_test_split(train_val,random_state = 2000,)
    # features
    X_train=training_data['statement']
    X_test=testing_data['statement']
    # GET LABELS
    Y_train=training_data['label'].values
    Y_test=testing_data['label'].values
    # GET FEATURES
    train_features,test_features,feature_transformer=extract_features(field,X_train,X_test,type=feature_rep)
    # INIT LOGISTIC REGRESSION CLASSIFIER
    logging.info("Training a Classification Model...")
    # scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    model=classifier.fit(train_features,Y_train)
    # GET PREDICTIONS
    predictions = model.predict(test_features)
    # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
    logging.info("Starting evaluation...")
    score = f1_score(Y_test,predictions)
    print(classification_report(Y_test,predictions))
    print(confusion_matrix(Y_test,predictions))
    logging.info("Done training and evaluation.")
    
    return model,feature_transformer,score


# Extract features for final training
def extract_final_features(field,training_data,type):
    """Extract features using different methods"""
    logging.info("Extracting features and creating vocabulary...")
    
    if "binary" in type:
        # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.95)
        cv.fit_transform(training_data.values)
        train_feature_set=cv.transform(training_data.values)
        return train_feature_set,cv
  
    elif "counts" in type:
        # COUNT BASED FEATURE REPRESENTATION
        cv= CountVectorizer(binary=False, max_df=0.95)
        cv.fit_transform(training_data.values)
        train_feature_set=cv.transform(training_data.values)
        return train_feature_set,cv
    
    else:    
        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit_transform(training_data.values)
        train_feature_set=tfidf_vectorizer.transform(training_data.values)
        return train_feature_set,tfidf_vectorizer


# Training best model on the entire dataset (train+val)
def train_final_model(classifier, train_val, field="statement",feature_rep="binary"):
    """
    Training the best classifier on entire dataset for the provided features.
    """
    logging.info("Starting model training...")    
    # features
    train_x=train_val['statement']
    # GET LABELS
    target=train_val['label'].values
    # GET FEATURES
    features,feature_transformer=extract_final_features(field,train_x,type=feature_rep)
    # INIT LOGISTIC REGRESSION CLASSIFIER
    logging.info("Training a Final Model...")
    # scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    model=classifier.fit(features,target)
    logging.info("Done training.")
    
    return model,feature_transformer