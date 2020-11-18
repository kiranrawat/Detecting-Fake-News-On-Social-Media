#!/usr/bin/env python
# coding: utf-8

# In[39]:


import Data_Preparation
import feature_selection
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold
import nltk
import nltk.corpus 


# ## LabelEncoding the target

# In[2]:


Encoder = LabelEncoder()
train_label = Encoder.fit_transform(Data_Preparation.train_news.label)
val_label = Encoder.fit_transform(Data_Preparation.val_news.label)
test_label = Encoder.fit_transform(Data_Preparation.test_news.label)


# In[3]:


display(Data_Preparation.train_news), display(Data_Preparation.test_news), display(Data_Preparation.val_news)


# In[6]:


train_label


# ## Naive Bayes Algorithm
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
# 
# 
# ## Metric to use? 
# 
# I need to minimize false positives (number of fake news predicted as real) as it can -vely impact people by misleadling them. For class 0 i.e. 'fake', recall should be high as well as precision. Because we want our model to perform well on both classes (real & fake). In short, we need to maximize f1-score.
# 
# ### Cases I considered to choose metric?
# 
# **1. If I just care to about maximizing recall of class 0 (fake) or minimizing false positives (FP)?**
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
# **2. Considering an extreme case, if all the news predicted are labelled as True. Even, fake news are predicted as True.**
# 
# Ex=>  TN = 0, FP = 553, TP = 714, FN =0
# In that case, TN will be 0, which led to Precision 0, Recall 0 and F1 = 0 for class 0 ('fake').
# 
# For class 1, Class1-Recall = TP / (TP + FN) = 1
# Class1-Precision = TP / (TP + FP) = 0.56
# 
# Hence, we care about model's performance in both classes i.e. precision and recall for both class 0 and class 1. 

# In[44]:


nb_clf_pipeline = Pipeline([('vect', feature_selection.count_vect),
                      ('tfidf', feature_selection.tfidf_transformer),
                      ('nb_clf', MultinomialNB()),
 ])
nb_clf_pipeline.fit(Data_Preparation.train_news['statement'], train_label)
predicted = nb_clf_pipeline.predict(Data_Preparation.test_news['statement'])
print(np.mean(predicted == test_label))
print(classification_report(test_label,predicted))
print(confusion_matrix(test_label,predicted))


# ## logistic regression
# 
# How hypothesis makes prediction in logistics regression?
# 
# This algorithm uses sigmoid function(g(z)). If we want to predict if y=1 or y=0.
# If estimated probability of y=1 is h(x)>=0.5 then the ouput is more likely to be "y=1" 
# but if  h(x) < 0.5, the output is more likely to be is "y=0".

# In[45]:


logR_pipeline = Pipeline([
        ('LogRCV',feature_selection.count_vect),
        ('LogR_clf',LogisticRegression())
        ])

logR_pipeline.fit(Data_Preparation.train_news['statement'],train_label)
predicted_LogR = logR_pipeline.predict(Data_Preparation.test_news['statement'])
print(np.mean(predicted_LogR == test_label))
print(classification_report(test_label,predicted_LogR))
print(confusion_matrix(test_label,predicted_LogR))


# ## SVM

# In[46]:


svm_pipeline = Pipeline([
        ('svmCV',feature_selection.count_vect),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(Data_Preparation.train_news['statement'],train_label)
predicted_svm = svm_pipeline.predict(Data_Preparation.test_news['statement'])
print(np.mean(predicted_svm == test_label))
print(classification_report(test_label,predicted_svm))
print(confusion_matrix(test_label,predicted_svm))


# ## Random Forest

# In[47]:


random_forest = Pipeline([
        ('rfCV',feature_selection.count_vect),
        ('rf_clf',RandomForestClassifier(n_estimators=200))
        ])
    
random_forest.fit(Data_Preparation.train_news['statement'],train_label)
predicted_rf = random_forest.predict(Data_Preparation.test_news['statement'])
print(np.mean(predicted_rf == test_label))
print(classification_report(test_label,predicted_rf))
print(confusion_matrix(test_label,predicted_rf))


# ## Merging train, val and test data for K-Fold

# In[28]:


frames = [Data_Preparation.train_news.drop('length', axis=1), Data_Preparation.val_news]
train_val = pd.concat(frames)
train_val


# In[29]:


train_val['label'].value_counts()


# In[35]:


train_val['label'] = Encoder.fit_transform(train_val['label'])


# In[36]:


train_val['label']


# So, we have merged all three datasets (train, test & val) together, so that we can run Naive Bayes with k-fold cross validation.

# ## K-fold cross validation

# In[37]:


# cross validation with cat boost classification
def apply_crossvalidation(classifier):

    k_fold = KFold(n_splits=5, shuffle=True)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for fold_n, (train_index, valid_index) in enumerate(k_fold.split(train_val['statement'], train_val['label'])):
        print(fold_n, len(train_index), len(valid_index))
        train_x = train_val['statement'].iloc[train_index]
        train_y = train_val['label'].iloc[train_index]
    
        valid_x = train_val['statement'].iloc[valid_index]
        valid_y = train_val['label'].iloc[valid_index]
    
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(valid_x)
        
        confusion += confusion_matrix(valid_y,predictions)
        score = f1_score(valid_y,predictions)
        scores.append(score)
        
    return (print('Total statements classified:', len(train_val['statement'])),
    print('Score:', sum(scores)/len(scores)),
    print('score length', len(scores)),
    print('Confusion matrix:'),
    print(confusion))


# In[40]:


apply_crossvalidation(nb_clf_pipeline)


# In[41]:


apply_crossvalidation(logR_pipeline)


# In[42]:


apply_crossvalidation(svm_pipeline)


# In[43]:


apply_crossvalidation(random_forest)

