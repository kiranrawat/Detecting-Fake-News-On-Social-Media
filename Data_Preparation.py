#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import nltk


# In[4]:


header_list = ["id", "label", "statement", "subject", "speaker", 
               "speaker_job", "speaker_state", "speaker_affiliation", 
               "barely_true_counts", "false_counts", "half_true_counts", 
               "mostly_true_counts", "pants_on_fire_counts", "context"]

train_data = pd.read_csv('liar_dataset/train.tsv', sep='\t',  names=header_list)
val_data = pd.read_csv('liar_dataset/valid.tsv', sep='\t',  names=header_list)
test_data = pd.read_csv('liar_dataset/test.tsv', sep='\t', names=header_list)


# In[5]:


#data observation
def data_obs():
    print("training dataset size:")
    print(train_data.shape)
    print(train_data.head(10))

    print("validation dataset size:")
    print(val_data.shape)
    print(val_data.head(10))
    
    print("test dataset size:")
    print(test_data.shape)
    print(test_data.head(10))


# 

# In[6]:


data_obs()


# In[7]:


#distribution of classes for prediction
def create_distribution(dataFile):
    
    return sn.countplot(x='label', data=dataFile, palette='hls')


# In[8]:


#by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
create_distribution(train_data)


# In[9]:


create_distribution(test_data)


# In[10]:


create_distribution(val_data)


# In[12]:


train_data.info(), test_data.info(), val_data.info()


# In[10]:


print(train_data['label'].value_counts())
print(val_data['label'].value_counts())
print(test_data['label'].value_counts())
# Mapping of lables
# True -- True
# Mostly-true -- True
# Half-true -- True
# Barely-true -- False
# False -- False
# Pants-fire -- False


# In[11]:


def map_lables(train,test,val):
    labels_dict = {'true': 'true','mostly-true': 'true',
                   'half-true':'true', 'false':'false', 
                   'barely-true':'false','pants-fire':'false'}
    
    train= train.replace({"label": labels_dict})[['label','statement']]
    test = test.replace({"label": labels_dict})[['label','statement']]
    val = val.replace({"label": labels_dict})[['label','statement']]
    display(train.head(3), test.head(3), val.head(3))
    
    return train, test, val  


# In[12]:


train_news, test_news, val_news = map_lables(train_data,test_data,val_data)


# In[13]:


#data integrity check (missing label values)
#none of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    
    print("Checking train data qualitites...")
    train_news.isnull().sum()
    train_news.info()
        
    print("check finished...")
    print()

    print("Checking test data qualitites...")
    test_news.isnull().sum()
    test_news.info()
    print("check finished...")
    print()
    
    print("Checking validation data qualitites...")
    val_news.isnull().sum()
    val_news.info()


# In[14]:


data_qualityCheck()


# In[15]:


#by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
print(create_distribution(train_news))


# In[16]:


create_distribution(test_news)


# In[17]:


create_distribution(val_news)


# In[12]:


# train_news.to_csv('train.csv', index=False)
# val_news.to_csv('val.csv', index=False)
# test_news.to_csv('test.csv', index=False)                          

