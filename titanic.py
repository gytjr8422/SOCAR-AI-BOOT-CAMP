#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer


# In[33]:


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# In[34]:


train.info()


# In[35]:


gender_labels = {'male':0, 'female':1}
train['Sex'] = train['Sex'].replace(gender_labels)
test['Sex'] = test['Sex'].replace(gender_labels)


# In[36]:


train


# In[37]:


train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[38]:


train.info()


# In[39]:


test.info()


# In[40]:


train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna('S')

test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# In[55]:


embarked_mapping = {'S':0, 'C':1, 'Q':2}
train['Embarked'] = train['Embarked'].replace(embarked_mapping)
test['Embarked'] = test['Embarked'].replace(embarked_mapping)


# In[56]:


train


# In[57]:


age = train['Age'].values.reshape(-1, 1)
fare = train['Fare'].values.reshape(-1, 1)

scaler = MinMaxScaler()
age_scaled = scaler.fit_transform(age)
fare_scaled = scaler.fit_transform(fare)
train['Age'] = age_scaled
train['Fare'] = fare_scaled


# In[58]:


train


# In[59]:


x_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']


# In[78]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=4)
rf.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(x_train, y_train)))


# In[82]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring='accuracy')
print("Cross Val Score: ", scores)
print("Mean: {:.3f}".format(scores.mean()))


# In[83]:


y_predict = rf.predict(test)


# In[84]:


y_predict


# In[ ]:




