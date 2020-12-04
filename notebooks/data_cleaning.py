#!/usr/bin/env python
# coding: utf-8

# # Import necessary dependencies and settings

# In[1]:


import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import nltk

import sys
sys.path.append("/Users/kiranrawat/Desktop/Personal Projects/Detecting-Fake-News-On-Social-Media/src")
from cleaning import *

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the Data

# In[2]:


header_list = ["id", "label", "statement", "subject", "speaker", 
               "speaker_job", "speaker_state", "speaker_affiliation", 
               "barely_true_counts", "false_counts", "half_true_counts", 
               "mostly_true_counts", "pants_on_fire_counts", "context"]

train_data = pd.read_csv('../data/raw/train.tsv', sep='\t',  names=header_list)
val_data = pd.read_csv('../data/raw/valid.tsv', sep='\t',  names=header_list)
test_data = pd.read_csv('../data/raw/test.tsv', sep='\t', names=header_list)


# # Dropping nonmandatory features

# In[3]:


print(train_data.columns.to_list())
# print(val_data.columns.to_list())
# print(test_data.columns.to_list())


# Training, Validation and test datasets contain 14 features such as id, label, statement and speaker, speaker_job, speaker_state etc.
# To build a fake news prediction model, we don't need speaker-related information as the speaker will not always be consistent. It will keep on changing. So, we are just keeping "statement" and "label".

# In[4]:


train = train_data[['label', 'statement']]
val = val_data[['label', 'statement']]
test = test_data[['label', 'statement']]


# # Data analysis

# In[5]:


#data observation
def data_obs():
    """
    Function to check the shape and first 5 rows of the datasets
    """
    print("training dataset size:")
    print(train.shape)
    print(train.head())

    print("validation dataset size:")
    print(val.shape)
    print(val.head())
    
    print("test dataset size:")
    print(test.shape)
    print(test.head())


# 

# In[6]:


data_obs()


# In[7]:


print(test.iloc[6]['statement'])


# # Data Quality Check

# **The info() method prints information about a DataFrame including the index dtype and columns, non-null values and memory usage.**

# In[8]:


#data integrity check (missing label values)
#none of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    
    print("Checking train data qualitites...")
    print(train.isnull().sum())
    print(train.info())
        
    print("check finished...")
    print()

    print("Checking test data qualitites...")
    print(test.isnull().sum())
    print(test.info())
    print("check finished...")
    print()
    
    print("Checking validation data qualitites...")
    print(val.isnull().sum())
    print(val.info())


# In[9]:


data_qualityCheck()


# There are not a single null values or missing data. We are good to go.

# # Label Distribution

# ## Labels by count

# In[10]:


len(set(train_data['label'].values))


# In[11]:


print(train_data['label'].value_counts())
print(val_data['label'].value_counts())
print(test_data['label'].value_counts())


# In[12]:


# #distribution of classes for prediction
# def create_distribution(dataFile):
    
#     return sn.countplot(x='label', data=dataFile, palette='hls')


# In[13]:


create_distribution(train)


# In[14]:


create_distribution(test)


# In[15]:


create_distribution(val)


# From the above distribution plots, it is evident, majority of the news articles are falling under 'half-true','mostly true' and 'false'lables.

# Using Pandas info() method, we see that there is not any null values in the datasets. 

# ## Mapping the lables into 'True' and 'False'
# 
# 1. True -- True
# 2. Mostly-true -- True
# 3. Half-true -- True
# 4. Barely-true -- False
# 5. False -- False
# 6. Pants-fire -- False

# In[16]:


# def map_lables(train,test,val):
#     labels_dict = {'true': 'true','mostly-true': 'true',
#                    'half-true':'true', 'false':'false', 
#                    'barely-true':'false','pants-fire':'false'}
    
#     train= train.replace({"label": labels_dict})[['label','statement']]
#     test = test.replace({"label": labels_dict})[['label','statement']]
#     val = val.replace({"label": labels_dict})[['label','statement']]
#     display(train.head(3), test.head(3), val.head(3))
    
#     return train, test, val  


# In[17]:


train_news, test_news, val_news = map_lables(train,test,val)


# In[18]:


print(create_distribution(train_news))


# In[19]:


create_distribution(test_news)


# In[20]:


create_distribution(val_news)


# By calling distribution funtion on the mapped datasets, we can see that training, test and valid data seems to be failry evenly distributed between the classes.

# ## Finding distribution of Statement lengths in News Articles

# In[21]:


train_line_lengths = [len(statement) for statement in train_news['statement']]
plt.plot(train_line_lengths)
plt.show()


# In[22]:


val_line_lengths = [len(statement) for statement in val_news['statement']]
plt.plot(val_line_lengths)
plt.show()


# In[23]:


test_line_lengths = [len(statement) for statement in test_news['statement']]
plt.plot(test_line_lengths)
plt.show()


# **From the distribution above, it is clear that there are some outliers i.e. statements with a quite high length. Lets get into more details and filter the statements with the length higher than 500.**

# In[24]:


train_news['len'] = [len(statement) for statement in train_news['statement']]


# In[25]:


train_news[(train_news['len'] > 500)] 


# In[26]:


train_news


# In[27]:


train_news.iloc[7550]['statement']


# In[28]:


test_news['len'] = [len(statement) for statement in test_news['statement']]


# ## Drop the rows with incorrect parsing.

# In[29]:


train_news = train_news[train_news['len'] < 500]
test_news = test_news[test_news['len'] < 500]


# **So, the data was not parsed correctly. For now, I will be just ignoring these rows. More data cleaning will be performed later in the process.** 

# ### Saving the processed version of the datasets.

# In[30]:


train_news.to_csv('../data/processed/train.csv', index=False)
val_news.to_csv('../data/processed/val.csv', index=False)
test_news.to_csv('../data/processed/test.csv', index=False)                          


# In[ ]:




