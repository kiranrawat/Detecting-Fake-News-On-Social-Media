# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

#Import necessary dependencies and settings
import seaborn as sn
import re
import string
from nltk.corpus import stopwords

#distribution of classes for prediction
def create_distribution(dataFile):
    """ check the data distribution
    """
    return sn.countplot(x='label', data=dataFile, palette='hls')

# Cleaning text
def process_text(text):
    """
    What will be covered:
    1. Lower case and remove special characters\whitespaces
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    """
    #1  
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    
    #2
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #3
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #4
    return clean_words

# Mapping the lables into 'True' and 'False'
def map_lables(train, test, val):
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
    
    return train, test, val  