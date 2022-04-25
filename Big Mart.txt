#!/usr/bin/env python
# coding: utf-8

# In[166]:


import pandas as pd
import numpy as np


# In[167]:


train_data = pd.read_csv(r'C:\Users\Olamiotan.000\Desktop\PWC\Big Mart Sales\train_kOBLwZA.csv')
test_data = pd.read_csv(r'C:\Users\Olamiotan.000\Desktop\PWC\Big Mart Sales\test_t02dQwI.csv')


# In[168]:


train_data.head()


# In[169]:


test_data.head()


# In[172]:


#view where there are null values
test_data.apply(lambda x: sum(x.isnull()))


# Apart from the Item_Identifier and Outlet_Identifier, every other column plays a role in predictiing the Item_Outlet_Sales

# In[173]:


#view contennts of the column Item_Fat_Content
test_data['Item_Fat_Content'].unique()


# In[174]:


# create a uniform name for LowFat and Regular Item_Fat_Content
train_data['Item_Fat_Content'].replace(['reg','low fat','LF'],['Regular','Low Fat','Low Fat'], inplace=True)
test_data['Item_Fat_Content'].replace(['reg','low fat','LF'],['Regular','Low Fat','Low Fat'], inplace=True)


# In[175]:


# creating new column est_year to get the number of years of store been s
train_data['est_years'] = train_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x) 
test_data['est_years'] = test_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)


# In[176]:


train_data['est_years'].head()


# In[177]:


train_data['Item_Type'].unique()


# In[178]:


train_data['Outlet_Type'].unique()


# In[212]:


train_data['Outlet_Size'].unique()


# In[216]:


train_data['Outlet_Location_Type'].unique()


# In[143]:


combined_data = [train_data, test_data]


# In[185]:


#code to capture Nan values
train_data[:] = np.nan_to_num(train_data)
test_data[:] = np.nan_to_num(test_data)


# In[144]:


col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']


# In[186]:


# handling catagorical variables
train_dummy = pd.get_dummies(train_data[:], columns = col, drop_first = True)
test_dummy = pd.get_dummies(test_data[:], columns = col,drop_first = True)


# In[180]:


new_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'est_years',
       'Item_Fat_Content_Regular', 'Item_Type_Breads', 'Item_Type_Breakfast',
       'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
       'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
       'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
       'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
       'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
       'Outlet_Size_Medium', 'Outlet_Size_Small',
       'Outlet_Location_Type_Tier 2', 'Outlet_Location_Type_Tier 3',
       'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
       'Outlet_Type_Supermarket Type3']


# In[187]:


#Sets the "X" and "Y" dataframe values (X = [feat_cols], Y = ['Item_Outlet_Sales']) using traindata
X = train_dummy[new_cols]
y = train_dummy['Item_Outlet_Sales']


# In[193]:


# splitting data as X_train and X_test
from sklearn.model_selection import train_test_split
#train AI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)


# In[195]:


# creating XGBoost model
from xgboost.sklearn import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)


# In[204]:


#Tests the Model
y_pred[1:10]


# In[197]:


# calculating Root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))


# In[198]:


#calling rsme
rmse


# In[202]:


# predicting on actual test data
X_t = test_dummy[new_cols]
y_result = xgb.predict(X_t)


# In[205]:


y_result


# In[211]:


#create a new column 'Item_Outlet_Sales to save predictions
test_dummy['Item_Outlet_Sales'] = y_result

prediction = pd.DataFrame(test_dummy, columns= ['Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales'])
export_csv = prediction.to_csv (r'C://Users/Olamiotan.000/Downloads/BigMart_Sales_Submission.csv', index = None, header=True) 
export_csv
print('Successful!')


# In[ ]:




