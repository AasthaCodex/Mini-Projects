#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# ABC is an online content sharing platform that enables users to create, upload and share the content in the form of videos. It includes videos from different genres like entertainment, education, sports, technology and so on. The maximum duration of video is 10 minutes.
# 
# Users can like, comment and share the videos on the platform. 
# 
# Based on the user’s interaction with the videos, engagement score is assigned to the video with respect to each user. Engagement score defines how engaging the content of the video is. 
# 
# Understanding the engagement score of the video improves the user’s interaction with the platform. It defines the type of content that is appealing to the user and engages the larger audience.
# 
# # Objective
# 
# The main objective of the problem is to develop the machine learning approach to predict the engagement score of the video on the user level.

# ## Importing Libraries 

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math


# In[3]:


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# ## Loading the training and test dataset

# In[4]:


train = pd.read_csv("train_0OECtn8.csv")


# In[5]:


test = pd.read_csv("test_N4clAbW.csv")


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


train.shape


# In[9]:


test.shape


# ## Checking missing values if any

# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# In[12]:


train.dtypes


# In[13]:


test.dtypes


# In[14]:


train.gender.value_counts()


# In[15]:


test.gender.value_counts()


# In[16]:


train.profession.value_counts()


# In[17]:


test.profession.value_counts()


# ## Exploratory Data Analysis

# In[18]:


sns.countplot(data = train, x = 'gender')


# In[19]:


sns.countplot(data = train, x='profession')


# In[20]:


plt.hist(train['age'], bins = 20)


# ## Converting categorical variables into numerical representations

# In[21]:


label_encoder = LabelEncoder()
train['gender'] = label_encoder.fit_transform(train['gender'])
train['profession'] = label_encoder.fit_transform(train['profession'])


# In[22]:


train.head()


# In[23]:


label_encoder = LabelEncoder()
test['gender'] = label_encoder.fit_transform(test['gender'])
test['profession'] = label_encoder.fit_transform(test['profession'])


# In[24]:


test.head()


# ## Dropping the unnecessary columns

# In[25]:


df_train = train.drop(['row_id'], axis = 1)


# In[26]:


df_train.head()


# In[27]:


df_test = test.drop(['row_id'], axis = 1)


# In[28]:


df_test.head()


# ## Split the data into features (X) and target variable (y)

# In[29]:


X = df_train.drop(['engagement_score'], axis=1)
y = df_train['engagement_score']


# In[30]:


X.head()


# In[31]:


y.head()


# ## Split the preprocessed data into training and validation sets

# In[32]:


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=25)


# ## Initialize and train the xgboost

# In[33]:


import xgboost as xg


# In[34]:


model = xg.XGBRegressor(n_estimators = 10, seed = 42)


# In[35]:


model.fit(X_train, y_train)


# ## Predict engagement scores for the validation set

# In[36]:


pred = model.predict(X_valid)


# ## Evaluate the model's performance using R-squared

# In[37]:


R2 = metrics.r2_score(y_valid, pred)
R2


# ## Improving Model's Performance

# In[38]:


model = xg.XGBRegressor(n_estimators = 3000, seed = 42)


# In[39]:


model.fit(X, y)


# ## Predict engagement scores for the test set

# In[40]:


prediction = model.predict(df_test)


# In[41]:


prediction


# In[42]:


sub = pd.DataFrame({'row_id': test.row_id,'engagement_score':prediction})


# In[43]:


sub.head()


# ## Converting dataframe to csv

# In[45]:


sub.to_csv('submission.csv', index=False)

