# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

#Import necessary dependencies and settings
import seaborn as sn
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#distribution of classes for prediction
def create_distribution(dataFile):
    """ 
    check the data distribution
    Args:
        dataFile (dataframe): Pandas dataframe (train/test/val)
    Returns:
        Distribution Plot with label on x-axis
    """
    return sn.countplot(x='label', data=dataFile, palette='hls')

# Cleaning text
def process_text(text):
    """
    Cleaning the text
    Args: 
        text (string): news statement
    Returns:
        list of clean text words
    """
    #1  remove special characters\whitespaces\digits
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    
    #2 Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #3 Lower case & Remove stopwords
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #4 get lemmatized tokens
    lemmatized_words = [lemmatizer.lemmatize(word) for word in clean_words if (len(word) > 2 and len(word) < 14)]

    return lemmatized_words

def process_text2(text):
    text1 = " ".join(text)
    return text1

# Mapping the lables into 'True' and 'False' with dictionary
def map_lables(train, test, val):
    """
    Args: 
        train (pandas.core.frame.DataFrame)): training data
        test (pandas.core.frame.DataFrame)): test data
        val (pandas.core.frame.DataFrame)): validation data
    Returns:
        label updated with either True or False

    How mapping is done: 
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