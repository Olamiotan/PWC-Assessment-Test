#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#importing csv files into dataframe
train_data = pd.read_csv(r'C:\Users\Olamiotan.000\Desktop\PWC\Big Mart Sales\train_kOBLwZA.csv')
test_data = pd.read_csv(r'C:\Users\Olamiotan.000\Desktop\PWC\Big Mart Sales\test_t02dQwI.csv')

#printing the first five rows to view dataset
train_data.head()
test_data.head()

#view where there are null values
test_data.apply(lambda x: sum(x.isnull()))


# Apart from the Item_Identifier and Outlet_Identifier, every other column plays a role in predictiing the Item_Outlet_Sales
#view contents of the column Item_Fat_Content
test_data['Item_Fat_Content'].unique()

# create a uniform name for LowFat and Regular Item_Fat_Content
train_data['Item_Fat_Content'].replace(['reg','low fat','LF'],['Regular','Low Fat','Low Fat'], inplace=True)
test_data['Item_Fat_Content'].replace(['reg','low fat','LF'],['Regular','Low Fat','Low Fat'], inplace=True)

# creating new column est_year to get the number of years of store been s
train_data['est_years'] = train_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x) 
test_data['est_years'] = test_data['Outlet_Establishment_Year'].apply(lambda x: 2013 - x)

#printing the first five rows to view column
train_data['est_years'].head()

#printig the unique values in these columns
train_data['Item_Type'].unique()
train_data['Outlet_Type'].unique()
train_data['Outlet_Size'].unique()
train_data['Outlet_Location_Type'].unique()

#combining the train and test datasets
combined_data = [train_data, test_data]

#code to capture Nan values
train_data[:] = np.nan_to_num(train_data)
test_data[:] = np.nan_to_num(test_data)

#adding new coumn names
col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# handling catagorical variables
train_dummy = pd.get_dummies(train_data[:], columns = col, drop_first = True)
test_dummy = pd.get_dummies(test_data[:], columns = col,drop_first = True)

#adding new column names/headers
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

#Sets the "X" and "Y" dataframe values (X = [feat_cols], Y = ['Item_Outlet_Sales']) using traindata
X = train_dummy[new_cols]
y = train_dummy['Item_Outlet_Sales']

#importing data model
from sklearn.model_selection import train_test_split

# splitting data as X_train and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)

# creating XGBoost model
from xgboost.sklearn import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

#Testing the Model
y_pred[1:10]

#calculating Root mean square error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))

#calling rsme
rmse

# predicting on actual test data
X_t = test_dummy[new_cols]
y_result = xgb.predict(X_t)

#callig /printing y_result
y_result

#create a new column 'Item_Outlet_Sales to save predictions
test_dummy['Item_Outlet_Sales'] = y_result
prediction = pd.DataFrame(test_dummy, columns= ['Item_Identifier', 'Outlet_Identifier','Item_Outlet_Sales'])

#exporting result into a csv file
export_csv = prediction.to_csv (r'C://Users/Olamiotan.000/Downloads/BigMart_Sales_Submission.csv', index = None, header=True) 
export_csv
print('Successful!')
