#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
import nltk
import nltk.corpus 
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
from gensim.models.word2vec import Word2Vec


# In[2]:


train_news = pd.read_csv('../data/processed/train.csv')
val_news = pd.read_csv('../data/processed/val.csv')
test_news = pd.read_csv('../data/processed/test.csv')


# In[3]:


print("====Label Distribution in Training Data ====")
print(train_news['label'].value_counts())
print("====Label Distribution in Validation Data ====")
print(val_news['label'].value_counts())
print("====Label Distribution in Test Data====")
print(test_news['label'].value_counts())


# By seeing the label's distribution, it seems like a balanced class. As number of 'True' and 'False' lables are kind of equally distributed in the dataset.

# In[4]:


train_news.groupby('label').describe()


# from above information, we know that:
# 
# 1. About 44% of the statements are classified as a True.
# 2. There are some duplicate messages, since the number of unique values lower than the count values of the text.

# In the next part, lext check the length of each text messages to see whether it is correlated with the text classified as a True or False.

# In[5]:


train_news['length'] = train_news['statement'].apply(len)


# In[6]:


train_news.hist(column='length',by='label',bins=60, figsize=(15,6))


# from above figure, we can see almost both True and False statements have length under 500.

# In[7]:


def process_text(text):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words


# In[8]:


train_news['statement'].apply(process_text).head()


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

# In[9]:


count_vect = CountVectorizer(analyzer=process_text)
X_train_counts = count_vect.fit_transform(train_news.statement)
X_train_counts.shape


# Here by doing ‘count_vect.fit_transform(train_news.statement)’, we are learning the vocabulary dictionary and it returns a Document-Term matrix. [n_samples, n_features]

# In[10]:


print(count_vect)
print(X_train_counts)


# In[11]:


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# In[12]:


print(X_train_tfidf)


# In[14]:


count_vect.get_feature_names()


# In[ ]:




