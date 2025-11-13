# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 11:56:49 2020
@author: Kiran Rawat
"""

# Import necessary dependencies and settings
import seaborn as sn
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# distribution of classes for prediction


def create_distribution(dataFile):
    """ 
    check the data distribution
    Args:
        dataFile (dataframe): Pandas dataframe (train/test/val)
    Returns:
        Distribution Plot with label on x-axis
    """
    return sn.countplot(x='label', data=dataFile, palette='hls')


def get_pos_tag(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    pos_dict = {
        'J': 'a',
        'N': 'n',
        'V': 'v',
        'R': 'r'
    }
    return pos_dict.get(tag, 'n')


# Cleaning text

def process_text(text):
    """
    Clean a text string and return a list of tokens.
    Optimized and more linguistically correct version.
    """
    # 1. lowercase
    text = text.lower()

    # 2. keep only letters and whitespace
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. tokenize (faster than split and safer)
    tokens = text.split()

    # 4. remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # 5. # POS-aware lemmatization
    tokens = [lemmatizer.lemmatize(t, get_pos_tag(t)) for t in tokens]

    return tokens


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
    labels_dict = {'true': 'true', 'mostly-true': 'true',
                   'half-true': 'true', 'false': 'false',
                   'barely-true': 'false', 'pants-fire': 'false'}

    train = train.replace({"label": labels_dict})[['label', 'statement']]
    test = test.replace({"label": labels_dict})[['label', 'statement']]
    val = val.replace({"label": labels_dict})[['label', 'statement']]

    return train, test, val
