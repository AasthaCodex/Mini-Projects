#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[2]:


train_data = pd.read_csv('train_BRCpofr_yY5tzKr.csv')
train_data.head()


# In[3]:


test_data = pd.read_csv('test_koRSKBP_Ua94SFA.csv')
test_data.head()


# In[4]:


train_data.shape


# In[5]:


test_data.shape


# In[6]:


train_data.describe()


# In[7]:


test_data.describe()


# In[8]:


train_data.isnull().sum()


# In[9]:


test_data.isnull().sum()


# In[10]:


train_data.nunique()


# In[11]:


test_data.nunique()


# In[12]:


train_data.columns


# In[13]:


train = train_data.drop('id', axis=1)


# In[14]:


test = test_data.drop('id', axis=1)


# In[15]:


train.gender.value_counts()


# In[16]:


train.area.value_counts()


# In[17]:


train.qualification.value_counts() 


# In[18]:


train.income.value_counts() 


# In[19]:


train.marital_status.value_counts()


# In[20]:


train.vintage.value_counts()


# In[21]:


train.num_policies.value_counts()


# In[22]:


train.policy.value_counts()


# In[23]:


train.type_of_policy.value_counts()


# In[24]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='gender')
plt.title(f"Frequency Distribution of Gender")
plt.show()


# In[25]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='area')
plt.title(f"Frequency Distribution of Area")
plt.show()


# In[26]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='qualification')
plt.title(f"Frequency Distribution of Qualification")
plt.show()


# In[27]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='income')
plt.title(f"Frequency Distribution of Income")
plt.show()


# In[28]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='marital_status')
plt.title(f"Frequency Distribution of Marital Status")
plt.show()


# In[29]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='policy')
plt.title(f"Frequency Distribution of Policy")
plt.show()


# In[30]:


plt.figure(figsize=(4, 3))
sns.countplot(data=train, x='type_of_policy')
plt.title(f"Frequency Distribution of Type of Policy")
plt.show()


# In[31]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='gender', y = 'cltv')
plt.show()


# In[32]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='area', y = 'cltv')
plt.show()


# In[33]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='qualification', y = 'cltv')
plt.show()


# In[34]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='income', y = 'cltv')
plt.show()


# In[35]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='marital_status', y = 'cltv')
plt.show()


# In[36]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='vintage', y = 'cltv')
plt.show()


# In[37]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='num_policies', y = 'cltv')
plt.show()


# In[38]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='policy', y = 'cltv')
plt.show()


# In[39]:


plt.figure(figsize=(4, 3))
sns.barplot(data=train, x='type_of_policy', y = 'cltv')
plt.show()


# In[40]:


plt.figure(figsize=(4, 3))
plt.boxplot(train['claim_amount'])
plt.xlabel('claim_amount')
plt.show()


# In[41]:


plt.figure(figsize=(4, 3))
plt.boxplot(train['cltv'])
plt.xlabel('cltv')
plt.show()


# In[42]:


plt.figure(figsize=(4, 3))
plt.boxplot(train['vintage'])
plt.xlabel('vintage')
plt.show()


# In[43]:


numerical_columns = ['vintage', 'claim_amount', 'num_policies', 'cltv']


# In[44]:


correlation_matrix = train[numerical_columns].corr()
plt.figure(figsize=(4, 3))
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()


# In[45]:


categorical_columns = ['gender', 'area', 'qualification', 'income', 'policy', 'type_of_policy']


# In[46]:


label_encoder = LabelEncoder()

for column in categorical_columns:
    train[column] = label_encoder.fit_transform(train[column])
    test[column] = label_encoder.fit_transform(test[column])


# In[47]:


train['num_policies'] = train['num_policies'].replace('More than 1', '2')
test['num_policies'] = test['num_policies'].replace('More than 1', '2')

train['num_policies'] = pd.to_numeric(train['num_policies'])
test['num_policies'] = pd.to_numeric(test['num_policies'])


# In[48]:


min_max_scaler = MinMaxScaler()

numerical_column = ['vintage', 'claim_amount', 'num_policies']

train[numerical_column] = min_max_scaler.fit_transform(train[numerical_column])
test[numerical_column] = min_max_scaler.transform(test[numerical_column])


# In[49]:


train.head()


# In[50]:


test.head()


# In[51]:


X = train.drop('cltv', axis=1)
y = train['cltv']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[52]:


from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error


# In[53]:


model = CatBoostRegressor()
model.fit(X_train, y_train)


# In[54]:


pred_train=model.fit(X_train,y_train).predict(X_train)
pred=model.fit(X_train,y_train).predict(X_val)


# In[55]:


r2 = r2_score(y_val, pred)
print("R Squared: ", r2)

train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
test_rmse = np.sqrt(mean_squared_error(y_val, pred))

print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)


# In[56]:


from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor


# In[57]:


param_grid = {
    'depth': [2, 4, 6],
    'learning_rate': [0.01, 0.1, 1],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5]
}


# In[58]:


model = CatBoostRegressor()


# In[59]:


random_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='r2', n_iter=10, random_state=42)


# In[60]:


random_search.fit(X_train, y_train)


# In[61]:


best_params = random_search.best_params_
best_score = random_search.best_score_


# In[62]:


print("Best Parameters:", best_params)
print("Best R-squared Score:", best_score)


# In[63]:


model=CatBoostRegressor(iterations=100 , depth= 4, l2_leaf_reg= 3, learning_rate = 0.1)


# In[64]:


model.fit(X_train, y_train)
pred_train=model.fit(X_train,y_train).predict(X_train)
pred=model.fit(X_train,y_train).predict(X_val)


# In[65]:


r2 = r2_score(y_val, pred)
print("R Squared: ", r2)

train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
test_rmse = np.sqrt(mean_squared_error(y_val, pred))

print("Train RMSE: ", train_rmse)
print("Test RMSE: ", test_rmse)


# In[66]:


model=CatBoostRegressor(iterations=100 , depth= 4, l2_leaf_reg= 3, learning_rate = 0.1)
fit=model.fit(X, y)


# In[67]:


pred=fit.predict(test)


# In[68]:


submission_df = pd.DataFrame({'id': test_data['id'], 'cltv': pred})


# In[69]:


submission_df.head()


# In[70]:


submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)

