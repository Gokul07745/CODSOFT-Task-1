#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


tn_ds = pd.read_csv('fraudTrain.csv')
tn_ds.head()


# In[4]:


tn_ds.tail()


# In[5]:


tn_ds.describe()


# In[6]:


print(tn_ds.isnull().sum())


# In[7]:


print(tn_ds.shape)


# In[8]:


ts_ds= pd.read_csv('fraudTest.csv')
ts_ds.head()


# In[9]:


ts_ds.describe()


# In[10]:


print(ts_ds.isnull().sum())


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=tn_ds)
plt.title('Distribution of Fraudulent Transactions')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()


# In[15]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='is_fraud', y='amt', data=tn_ds)
plt.title('Transaction Amount vs. Fraud')
plt.xlabel('Is Fraud')
plt.ylabel('Transaction Amount')
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
sns.countplot(x='gender', hue='is_fraud', data=tn_ds)
plt.title('Distribution of Gender by Fraud')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Is Fraud')
plt.show()


# In[17]:


plt.figure(figsize=(12, 6))
sns.countplot(x='category', hue='is_fraud', data=tn_ds)
plt.title('Distribution of Categories by Fraud')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha="right")
plt.legend(title='Is Fraud')
plt.show()


# In[21]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[24]:


# Encode categorical variables
encoder = OneHotEncoder(drop='first')
categorical_cols = ['gender', 'category', 'state']
encoded_train_features = encoder.fit_transform(tn_ds[categorical_cols]).toarray()
encoded_test_features = encoder.transform(ts_ds[categorical_cols]).toarray()


# In[25]:


scaler = StandardScaler()
numerical_cols = ['amt', 'lat', 'long','city_pop', 'unix_time', 'merch_lat', 'merch_long']
scaled_tn_features = scaler.fit_transform(tn_ds[numerical_cols])
scaled_ts_features = scaler.transform(ts_ds[numerical_cols])


# In[27]:


final_train_features = pd.concat([pd.DataFrame(encoded_train_features), pd.DataFrame(scaled_tn_features)], axis=1)
final_test_features = pd.concat([pd.DataFrame(encoded_test_features), pd.DataFrame(scaled_ts_features)], axis=1)


# In[29]:


train_target = tn_ds['is_fraud']
test_target = ts_ds['is_fraud']


# In[30]:


smote = SMOTE(random_state=36)

x_train_resample, y_train_resample = smote.fit_resample(final_train_features, train_target)


# In[31]:


print('Current length of the training set: ', len(y_train_resample))


# In[32]:


plt.figure(figsize=(8, 6))
sns.countplot(x=y_train_resample)
plt.title('Distribution of Fraudulent Transactions')
plt.xlabel('Is Fraud')
plt.ylabel('Count')
plt.show()


# In[35]:


X_shuffled, y_shuffled = shuffle(x_train_resample, y_train_resample, random_state=42)


# In[36]:


x_train, x_validation, y_train, y_validation = train_test_split(X_shuffled, y_shuffled, test_size=0.5)


# In[37]:


x_train_copy = x_train
y_train_copy = y_train

x_train = x_train[:10000]
y_train = y_train[:10000]


# In[40]:


lg_model = LogisticRegression(max_iter=1000)
lg_model.fit(x_train, y_train)

# Make predictions on test data
lg_predictions = lg_model.predict(x_validation)

# Calculate evaluation metrics on test data
lg_accuracy = accuracy_score(y_validation, lg_predictions)


# Print evaluation metrics with 3 decimal places, multiplied by 100
print("Logistic Regression Accuracy: {:.3f}%".format(lg_accuracy * 100))


# In[ ]:




