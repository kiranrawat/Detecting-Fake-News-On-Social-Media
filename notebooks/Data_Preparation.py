#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sn
import numpy as np
import nltk


# In[4]:


header_list = ["id", "label", "statement", "subject", "speaker", 
               "speaker_job", "speaker_state", "speaker_affiliation", 
               "barely_true_counts", "false_counts", "half_true_counts", 
               "mostly_true_counts", "pants_on_fire_counts", "context"]

train_data = pd.read_csv('../data/raw/train.tsv', sep='\t',  names=header_list)
val_data = pd.read_csv('../data/raw/valid.tsv', sep='\t',  names=header_list)
test_data = pd.read_csv('../data/raw/test.tsv', sep='\t', names=header_list)


# ## Data analysis

# In[5]:


#data observation
def data_obs():
    """
    Function to check the shape and first 5 rows of the datasets
    """
    print("training dataset size:")
    print(train_data.shape)
    print(train_data.head())

    print("validation dataset size:")
    print(val_data.shape)
    print(val_data.head())
    
    print("test dataset size:")
    print(test_data.shape)
    print(test_data.head())


# 

# In[6]:


data_obs()


# ### Data Quality

# **The info() method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.**

# In[15]:


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


# In[16]:


data_qualityCheck()


# ## Category Distribution

# ### Category by count

# In[17]:


len(set(train_data['label'].values))


# In[18]:


print(train_data['label'].value_counts())
print(val_data['label'].value_counts())
print(test_data['label'].value_counts())


# In[19]:


#distribution of classes for prediction
def create_distribution(dataFile):
    
    return sn.countplot(x='label', data=dataFile, palette='hls')


# In[20]:


create_distribution(train_data)


# In[21]:


create_distribution(test_data)


# In[22]:


create_distribution(val_data)


# From the above distribution plots, it is evident, majority of the news articles are falling under 'half-true','mostly true' and 'false'lables.

# Using Pandas info() method, we see that there is not any null values in the datasets. 

# ### Mapping the lables into 'True' and 'False'
# 
# 1. True -- True
# 2. Mostly-true -- True
# 3. Half-true -- True
# 4. Barely-true -- False
# 5. False -- False
# 6. Pants-fire -- False

# In[23]:


def map_lables(train,test,val):
    labels_dict = {'true': 'true','mostly-true': 'true',
                   'half-true':'true', 'false':'false', 
                   'barely-true':'false','pants-fire':'false'}
    
    train= train.replace({"label": labels_dict})[['label','statement']]
    test = test.replace({"label": labels_dict})[['label','statement']]
    val = val.replace({"label": labels_dict})[['label','statement']]
    display(train.head(3), test.head(3), val.head(3))
    
    return train, test, val  


# In[24]:


train_news, test_news, val_news = map_lables(train_data,test_data,val_data)


# In[25]:


print(create_distribution(train_news))


# In[26]:


create_distribution(test_news)


# In[27]:


create_distribution(val_news)


# By calling distribution funtion on the mapped datasets, we can see that training, test and valid data seems to be failry evenly distributed between the classes.

# ### Saving the processed version of the datasets.

# In[20]:


train_news.to_csv('../data/processed/train.csv', index=False)
val_news.to_csv('../data/processed/val.csv', index=False)
test_news.to_csv('../data/processed/test.csv', index=False)                          

