#!/usr/bin/env python
# coding: utf-8

# # Import necessary dependencies and settings

# In[18]:


import pandas as pd
import numpy as np
import string
import logging
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper

import nltk
from nltk.corpus import stopwords

#import python scripts to use user_defined functions
import sys
sys.path.append("/Users/kiranrawat/Desktop/Personal Projects/Detecting-Fake-News-On-Social-Media/src")
from cleaning import process_text
from training import extract_features, train_model, extract_final_features, train_final_model  
from prediction import get_predictions


# # Loading the Data

# In[9]:


train_news = pd.read_csv('../data/processed/train.csv').drop('len', axis=1)
val_news = pd.read_csv('../data/processed/val.csv')
test_news = pd.read_csv('../data/processed/test.csv').drop('len', axis=1)


# In[10]:


train_news.head()


# In[13]:


display(train_news), display(val_news), display(test_news)


# ## Merging train & val data for K-Fold

# In[14]:


"""
Merging the training and validation data together, so that I can peroform k-fold cross validation 
and shuffle the data to reduce the bias.
"""
labelEncoder = LabelEncoder()
frames = [train_news, val_news]
train_val = pd.concat(frames)
train_val['label'].value_counts()
train_val['label'] = labelEncoder.fit_transform(train_val['label'])


# In[15]:


train_val


# ## Feature Weighting
# 
# Not all words are equally important to a particular document / category. For example, while words like ‘murder’, ‘knife’ and ‘abduction’ are important to a crime related document, words like ‘news’ and ‘reporter’ may not be quite as important. 
# 
# ### Binary Weighting
# The most basic form of feature weighting, is binary weighting. Where if a word is present in a document, the weight is ‘1’ and if the word is absent the weight is ‘0’. 
# 
# ### CountVectorizer
# 
# It Convert a collection of text documents to a matrix of token counts.
# 
# 
# ### Tfidf Weighting 
# 
# TF-IDF weighting where words that are unique to a particular document would have higher weights compared to words that are used commonly across documents. 
# 
# 1. TF (Term Frequency): The number of times a word appears in a document divded by the total number of words in the document. Every document has its own term frequency.
# 
# 2. IDF (Inverse Data Frequency): The log of the number of documents divided by the number of documents that contain the word w. Inverse data frequency determines the weight of rare words across all documents in the corpus.
# 
# 3. Lastly, the TF-IDF is simply the TF multiplied by IDF.

# ## Metric
# 
# I need to minimize false positives (number of fake news predicted as real news) as it can be very misleadling . For class 0 i.e. 'fake', recall should be high as well as precision. Because we want our model to perform well on both classes (real & fake). In short, we need to maximize f1-score.
# 
# ### Cases I considered to choose the right metric
# 
# **1. Maximizing recall of class 0 (fake) or minimizing false positives(FP)?**
# Well, in extreme case, what if all the news predicted by model are labelled as 'fake'. Recall will still be 1, but overall model is really bad i.e. not able to predict class 1 ('real'). 
# 
# Ex=> TN = 553, FP = 0, TP = 0, FN = 714
# 
# Class0-Recall = TN / (TN + FP) = 1
# Class0-Precision = TN / (TN + FN) = 0.43
# 
# F1-Score = 2 * Class0-Recall * Class0-Precision/(Class0-Recall + Class0-Precision) = 0.60
# 
# Recall, Precision and F1-score for class 1 will be 0.
# 
# **2. Considering an extreme case, if all the news classified as True (Even, fake news are predicted as True).**
# 
# Ex=>  TN = 0, FP = 553, TP = 714, FN =0
# In that case, TN will be 0, which led to Precision 0, Recall 0 and F1 = 0 for class 0 ('fake').
# 
# For class 1, Class1-Recall = TP / (TP + FN) = 1
# Class1-Precision = TP / (TP + FP) = 0.56

# ## Model Training

# ## Text Classification Algorithms
# 
# 1. Naive Bayes (NB)
# 2. Logistics Regression
# 3. SVM
# 4. Random Forest

# ## Naive Bayes
# 
# Well, when assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data. An advantage of naive Bayes is that it only requires a small number of training data to estimate the parameters necessary for classification. 
# 
# Bayes’ Theorem provides a way that we can calculate the probability of a piece of data belonging to a given class, given our prior knowledge. Bayes’ Theorem is stated as:
# 
# P(class|data) = (P(data|class) * P(class)) / P(data)
# 
# Where P(class|data) is the probability of class given the provided data.
# 
# Naive Bayes is a classification algorithm for binary (two-class) and multiclass classification problems. It is called Naive Bayes or idiot Bayes because the calculations of the probabilities for each class are simplified to make their calculations tractable.
# 
# Rather than attempting to calculate the probabilities of each attribute value, they are assumed to be conditionally independent given the class value.
# 
# This is a very strong assumption that is most unlikely in real data, i.e. that the attributes do not interact. Nevertheless, the approach performs surprisingly well on data where this assumption does not hold.
# 
# ### Multinomial NB
# 
# The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work

# ### Train Models with Different Types of Features

# In[17]:


# model,transformer,score,confusion,report=train_model(nb_clf, train_val,field=field,feature_rep=feature_rep)
# print("\nF1-score={0}; confusion={1}; classification_report={2}".format(score,confusion,report))
field='statement'
feature_reps=['binary','counts','tfidf']
nb_results=[]
nb_clf = MultinomialNB()
for feature_rep in feature_reps:
        print(f'Model - {feature_rep} features with statement')
        nb_model,transformer,score=train_model(nb_clf,train_val,field=field,feature_rep=feature_rep)
        nb_results.append([field,feature_rep,score])


# ### Naive Bayes Results of Various Models

# In[19]:


nb_df_results=pd.DataFrame(nb_results,columns=['text_fields','feature_representation','f1-score'])
nb_df_results.sort_values(by=['f1-score'],ascending=False)


# ## logistic regression
# 
# The underlying algorithm is also fairly easy to understand. More importantly, in the NLP world, it’s generally accepted that Logistic Regression is a great starter algorithm for text related classification (https://web.stanford.edu/~jurafsky/slp3/5.pdf). 
# 
# **How hypothesis makes prediction in logistics regression?**
# 
# This algorithm uses sigmoid function(g(z)). If we want to predict y=1 or y=0.
# If estimated probability of y=1 is h(x)>=0.5 then the ouput is more likely to be "y=1" 
# but if  h(x) < 0.5, the output is more likely to be is "y=0".

# ### Train Models with Different Types of Features¶

# In[21]:


field='statement'
feature_reps=['binary','counts','tfidf']
lr_results=[]
LogR_clf = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)

for feature_rep in feature_reps:
        print(f'Model - {feature_rep} features with statement')
        lr_model,transformer,score=train_model(LogR_clf,train_val,field=field,feature_rep=feature_rep)
        lr_results.append([field,feature_rep,score])


# ### Logistics Regression Results of Various Models

# In[22]:


lr_df_results=pd.DataFrame(lr_results,columns=['text_fields','feature_representation','f1-score'])
lr_df_results.sort_values(by=['f1-score'],ascending=False)


# Here you see how the performance of logistics model is improved using tfidf over counts and binary weightning.

# ## SVM
# 
# Support vector machines is an algorithm that determines the best decision boundary between vectors that belong to a given group (or category) and vectors that do not belong to it. That’s it. It can be applied to any kind of vectors which encode any kind of data. This means that in order to leverage the power of svm text classification, texts have to be transformed into vectors.
# 
# So, when SVM determines the decision boundary we mentioned above, SVM decides where to draw the best “line” (or the best hyperplane) that divides the space into two subspaces: one for the vectors which belong to the given category and one for the vectors which do not belong to it.

# ### Train Models with Different Types of Features¶

# In[23]:


field='statement'
feature_reps=['binary','counts','tfidf']
svm_results=[]
svm_clf = svm.LinearSVC()

for feature_rep in feature_reps:
        print(f'SVM Model - {feature_rep} features with statement')
        svm_model,transformer,score=train_model(svm_clf,train_val,field=field,feature_rep=feature_rep)
        svm_results.append([field,feature_rep,score])


# ### SVM Results of Various Models

# In[24]:


svm_df_results=pd.DataFrame(svm_results,columns=['text_fields','feature_representation','f1-score'])
svm_df_results.sort_values(by=['f1-score'],ascending=False)


# ## Random Forest
# 
# Given the nature of random forests (a bagging decision tree), it is true that you may come up with a rather weak classifier, especially if only a couple of features are truly significant to determine the outcome.
# 
# However, keep in mind that in the case of text classification, a preprocessing phase is required to get either your TF or TF-IDF matrix, through which you have already made a selection of pertinent features. Potentially, all features are relevant in this matrix, so the random forest may be performant when you predict your outcome. (source: https://stats.stackexchange.com/questions/343954/random-forest-short-text-classification)

# ### Train Models with Different Types of Features¶

# In[25]:


field='statement'
feature_reps=['binary','counts','tfidf']
rf_results=[]
rf_clf = RandomForestClassifier(n_estimators=1000)

for feature_rep in feature_reps:
        rf_model,transformer,score=train_model(rf_clf,train_val,field=field,feature_rep=feature_rep)
        rf_results.append([field,feature_rep,score])


# ### RF Results of Various Models¶

# In[26]:


rf_df_results=pd.DataFrame(rf_results,columns=['text_fields','feature_representation','f1-score'])
rf_df_results.sort_values(by=['f1-score'],ascending=False)


# ## K-fold cross validation
# 
# With K-fold cross validation, you are testing how well your model is able to get trained by some data and then predict data it hasn't seen. We use cross validation for this because if you train using all the data you have, you have none left for testing. You could do this once, say by using 80% of the data to train and 20% to test, but what if the 20% you happened to pick to test happens to contain a bunch of points that are particularly easy (or particularly hard) to predict? We will not have come up with the best estimate possible of the models ability to learn and predict.

# In[30]:


#User defined functon for K-Fold cross validatoin
def apply_kfold(classifier,train_val,field,feature_rep):
    """
    K-fold cross validation on the the data
    """
    k_fold = KFold(n_splits=5, shuffle=True)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for fold_n, (train_index, valid_index) in enumerate(k_fold.split(train_val['statement'], train_val['label'])):
        print(fold_n, len(train_index), len(valid_index))
        train_x = train_val['statement'].iloc[train_index]
        train_y = train_val['label'].iloc[train_index]
    
        valid_x = train_val['statement'].iloc[valid_index]
        valid_y = train_val['label'].iloc[valid_index]
        
        # GET FEATURES
        train_features,val_features,feature_transformer=extract_features(field,train_x,valid_x,type=feature_rep)
        
        # INIT CLASSIFIER
        logging.info("Training a Classification Model...")
        classifier.fit(train_features, train_y)
        predictions = classifier.predict(val_features)
        
        confusion += confusion_matrix(valid_y,predictions)
        score = f1_score(valid_y,predictions)
        scores.append(score)
        
    return (print('Total statements classified:', len(train_val['statement'])),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))


# ## Grid Search Hyperparameters

# In[31]:


# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold
# from skopt import BayesSearchCV

# field='statement'
# feature_reps=['binary','counts','tfidf']
# # GET FEATURES
# train_features,feature_transformer=extract_final_features('statement',train_val['statement'],type='binary')
    
# # define search space
# params = dict()
# params['C'] = (1e-6, 100.0, 'log-uniform')
# params['gamma'] = (1e-6, 100.0, 'log-uniform')
# params['degree'] = (1,5)
# params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
# # define evaluation
# cv = StratifiedKFold(n_splits=5, random_state=1)
# # define the search
# search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
# # perform the search
# search.fit(train_features, train_val['label'])
# # report the best result
# print(search.best_score_)
# print(search.best_params_)


# ## Naive Bayes with K-fold cross validation

# In[32]:


field='statement'
feature_reps=['binary','counts','tfidf']
nb_results=[]
nb_clf = MultinomialNB()
for feature_rep in feature_reps:
        print(f'Model - {feature_rep} features with statement')
        apply_kfold(nb_clf,train_val,field=field,feature_rep=feature_rep)


# ## Logistics Regression with K-fold cross Validation

# In[33]:


field='statement'
feature_reps=['binary','counts','tfidf']
LogR_clf = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)

for feature_rep in feature_reps:
        print(f'Model - {feature_rep} features with statement')
        apply_kfold(LogR_clf,train_val,field=field,feature_rep=feature_rep)


# ## SVM with K-fold cross Validation

# In[ ]:


field='statement'
feature_reps=['binary','counts','tfidf']
svm_clf = svm.LinearSVC()

for feature_rep in feature_reps:
        print(f'Model - {feature_rep} features with statement')
        apply_kfold(svm_clf,train_val,field=field,feature_rep=feature_rep)


# ## RF with K-fold cross Validation

# In[ ]:


field='statement'
feature_reps=['binary','counts','tfidf']
rf_clf = RandomForestClassifier(n_estimators=1000)

for feature_rep in feature_reps:
        print(f'Model - {feature_rep} features with statement')
        apply_kfold(rf_clf,train_val,field=field,feature_rep=feature_rep)


# ## Best Model Selection

# """
# Out of all the models fitted, we would take 2 best performing model. we would call them candidate models
# from the confusion matrix, we can see that logistic regression and SVM (with either binary or tfidf features) are better performing 
# in terms of precision and recall (take a look into false positive and true negative counts which appeares
# to be low compared to rest of the models).
# 
# Using k-fold cross validation, we see the performance of the models on the entire dataset. And, the model's aren't performing well. We can apply other features to improve the performance, and grid-search can also help us to find best parameters to improve the perfromance.
# """

# In[ ]:


field='statement'
LogR_clf_final = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
lr_final_model,transformer=train_final_model(LogR_clf_final,train_val,field=field,feature_rep='counts')


# ## Check predictions on unseen data

# In[ ]:


test_features=transformer.transform(["When asked by a reporter whether hes at the center of a criminal scheme to violate campaign laws, Gov. Scott Walker nodded yes."])
ouput = get_predictions(lr_final_model,test_features)
ouput[0] # correctly predicted


# In[ ]:


test_features=transformer.transform(["Says John McCain has done nothing to help the vets."])
ouput = get_predictions(lr_final_model,test_features)
ouput[0] # false news predicted as false. #correct prediction


# In[ ]:


test_features=transformer.transform(["Says that Tennessee law requires that schools receive half of proceeds -- $31 million per year -- from a half-cent increase in the Shelby County sales tax."])
ouput = get_predictions(lr_final_model,test_features)
ouput[0] # True news predicted as True. #correct prediction


# ## Save Model for Future Use

# In[ ]:


import pickle

model_path="../models/lr_final_model.pkl"
transformer_path="../models/transformer.pkl"

# we need to save both the transformer -> to encode a document and the model itself to make predictions based on the weight vectors 
pickle.dump(lr_final_model,open(model_path, 'wb'))
pickle.dump(transformer,open(transformer_path,'wb'))


# In[ ]:




