#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np


# In[74]:


train_data = pd.read_csv(r'C:\Users\Dupe\Downloads\Loan Prediction\train_ctrUa4K.csv')
test_data = pd.read_csv(r'C:\Users\Dupe\Downloads\Loan Prediction\test_lAUu6dG.csv')


# In[75]:


train_data.head()


# In[41]:


test_data.head()


# In[78]:


#view where there are null values
train_data.apply(lambda x: sum(x.isnull()))


# In[79]:


train_data.shape


# In[43]:


test_data.apply(lambda x: sum(x.isnull()))


# In[80]:


test_data.shape


# In[81]:


#view contennts of the column 
train_data['Gender'].unique()


# In[82]:


train_data.describe()


# In[132]:


#Data Cleaning and filling missing values
train_data['Gender'] = train_data['Gender'].fillna('Male')
train_data['Married'] = train_data['Married'].fillna('Yes')
train_data['Dependents'] = train_data['Dependents'].fillna('0')
train_data['Self_Employed'] = train_data['Self_Employed'].fillna('No')
train_data['LoanAmount'] = train_data['LoanAmount'].fillna(train_data['LoanAmount'] .mean())
train_data['Loan_Amount_Term']  = train_data['Loan_Amount_Term'].fillna(360.0)
train_data['Credit_History'] = train_data['Credit_History'] .fillna(1.0)


# In[84]:


# Splitting traing data
X = train_data.iloc[:, 1: 12].values
y = train_data.iloc[:, 12].values


# In[85]:


X


# In[86]:


y


# In[171]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)


# In[135]:


X_train


# In[97]:


y_train


# In[172]:


# Using encoding categorical data/columns for the train data

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

for i in range(0, 5):
    X_train[:,i] = labelencoder_X.fit_transform(X_train[:,i])

X_train[:,10] = labelencoder_X.fit_transform(X_train[:,10])


# In[173]:


# Encoding the Dependent Variable for the train data
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)


# In[174]:


#  Using encoding categorical data/columns for the test data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
for i in range(0, 5):
    X_test[:,i] = labelencoder_X.fit_transform(X_test[:,i])
X_test[:,10] = labelencoder_X.fit_transform(X_test[:,10])
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y_test = labelencoder_y.fit_transform(y_test)


# In[175]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[176]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs')
classifier.fit(X_train, y_train)


# In[177]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[159]:


y_pred.shape


# In[178]:


# Measuring Accuracy
from sklearn import metrics
#Print accuracy
accuracy = metrics.accuracy_score(y_pred, y_test)
print ("The Percentage Accuracy using Logistic Regression: %s" % "{0:.3%}".format(accuracy))


# In[180]:


#create a new column 'Loan_Satus to save predictions

test_data['Loan_Status'] = y_pred

loan_predict = pd.DataFrame(test_data, columns= ['Loan_Id', 'Loan_Status'])
export_csv = loan_predict.to_csv (r'C://Users/Olamiotan.000/Downloads/loan_prediction_submission.csv', index = None, header=True) 
export_csv
print('Successful!')


# In[ ]:




