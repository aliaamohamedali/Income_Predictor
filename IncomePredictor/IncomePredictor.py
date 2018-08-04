
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import category_encoders as ce


# In[2]:


col_names = ['age', 'work-class', 'fnlwgt', 'education', 'education-num', 'marital-status', 
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
            'hrs-per-week', 'native-country', 'income']

data = pd.read_csv('adult.data.csv', header = None, names = col_names, na_values = " ?")

print(data.shape)
data.head(n = 10)


# In[3]:


## Search for NAN values:
print(data[data.isnull().any(axis=1)].count())
print(data.isnull().values.sum())
data = data.dropna(axis = 0)
print(data.shape)


# # Data Dictionary:
# age: continuous.
# 
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# 
# fnlwgt: continuous.
# 
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# 
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# 
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# 
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# 
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# 
# sex: Female, Male.
# 
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# 
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 

# # Handling Categorical Data

# In[4]:


data.dtypes


# In[5]:


## first the simplest ones
# Income: map >50K: 1, <=50K: 0
data['income'] = data['income'].map({' >50K': 1, ' <=50K': 0})
# data['income'].astype('int32')
# Sex: map Male: 1, Female: 0
data['sex'] = data['sex'].map({' Male': 1, ' Female': 0})
# data['sex'].astype('int32')
data.head(n = 10)


# In[6]:


##  Merge Never-worked & Without pay
data['work-class'] = data['work-class'].replace([' Without-pay', ' Never-worked'], 'Unpayed')
print(data['work-class'].value_counts().count())
data['work-class'].unique()


# In[7]:


## Not many different categories so will use Label Encoding
labels = data['work-class'].astype('category').cat.categories.tolist()
mapping = {'work-class': {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}

data.replace(mapping, inplace = True)

data.head(n = 10)


# In[8]:


data = data.drop(columns = ['education-num'], axis = 1)
print(data['education'].value_counts())

data.head(n = 2)


# In[9]:


data['education'] = data['education'].replace([' 10th', ' 11th', ' 12th'], 'HS-Student')
data['education'] = data['education'].replace([' 7th-8th', ' 9th'], 'Mid-Student')
data['education'] = data['education'].replace([' 5th-6th', ' 1st-4th'], 'Elem-Student')

print(data['education'].value_counts())


# In[10]:


labels = data['education'].astype('category').cat.categories.tolist()
mapping = {'education': {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}

data.replace(mapping, inplace = True)

data.head(n = 10)


# In[11]:


labels = data['marital-status'].astype('category').cat.categories.tolist()
mapping = {'marital-status': {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}
data.replace(mapping, inplace = True)

labels = data['relationship'].astype('category').cat.categories.tolist()
mapping = {'relationship': {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}
data.replace(mapping, inplace = True)

labels = data['race'].astype('category').cat.categories.tolist()
mapping = {'race': {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}
data.replace(mapping, inplace = True)

data.head(n = 10)


# In[12]:


## Occupation & Nativity have many categories > Binary Encode
encoder = ce.BinaryEncoder(cols = ['occupation', 'native-country'])
data = encoder.fit_transform(data)

data.head(n = 10)


# # The Model

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# In[14]:


labels = data['income']
features = data.drop(columns = ['income'], axis = 1)

labels.head(n = 5)
features.head(n = 5)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 2)
print(X_train.shape)
print(X_test.shape)


# In[16]:


model = XGBClassifier()
model.fit(X_train, y_train)
print(model)


# In[17]:


y_hat_train = model.predict(X_train)
train_pred = [round(value) for value in y_hat_train]

train_accuracy = accuracy_score(y_train, train_pred)
print('Train Accuracy: ', train_accuracy)


# In[18]:


y_hat_test = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_hat_test)
print('Test Accuracy: ', test_accuracy)

