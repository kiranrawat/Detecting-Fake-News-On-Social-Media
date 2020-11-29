# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

# import libraries
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import nltk

# Read CSV
header_list = ["id", "label", "statement", "subject", "speaker", 
               "speaker_job", "speaker_state", "speaker_affiliation", 
               "barely_true_counts", "false_counts", "half_true_counts", 
               "mostly_true_counts", "pants_on_fire_counts", "context"]

train_data = pd.read_csv('../data/raw/train.tsv', sep='\t',  names=header_list)
val_data = pd.read_csv('../data/raw/valid.tsv', sep='\t',  names=header_list)
test_data = pd.read_csv('../data/raw/test.tsv', sep='\t', names=header_list)


#data integrity check (missing label values)
#none of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    
    print("Checking train data qualitites...")
    print(train_data.isnull().sum())
    print(train_data.info())
        
    print("check finished...")
    print()

    print("Checking test data qualitites...")
    print(test_data.isnull().sum())
    print(test_data.info())
    print("check finished...")
    print()
    
    print("Checking validation data qualitites...")
    print(val_data.isnull().sum())
    print(val_data.info())

#check the data by calling below function
#data_qualityCheck()

#distribution of classes for prediction
def create_distribution(dataFile):
    
    return sn.countplot(x='label', data=dataFile, palette='hls')

# create_distribution(train_data)
# create_distribution(test_data)
# create_distribution(val_data)

# Mapping the lables into 'True' and 'False'
def map_lables(self, train, test, val):
     """ 
    1. True -- True
    2. Mostly-true -- True
    3. Half-true -- True
    4. Barely-true -- False
    5. False -- False
    6. Pants-fire -- False
    """
    labels_dict = {'true': 'true','mostly-true': 'true',
               'half-true':'true', 'false':'false', 
               'barely-true':'false','pants-fire':'false'}

    train= train.replace({"label": labels_dict})[['label','statement']]
    test = test.replace({"label": labels_dict})[['label','statement']]
    val = val.replace({"label": labels_dict})[['label','statement']]
    display(train.head(3), test.head(3), val.head(3))
    
    return train, test, val  
     
train_news, test_news, val_news = map_lables(train_data,test_data,val_data)
       
