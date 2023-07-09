#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In the hands-on exercises, here i  am using  data about home prices in Melbourne, Australia. 
# 
# The example (Melbourne) data is at the file path "C:\Users\User\Downloads\data_home_prices.csv".
# 
# We load and explore the data with the following commands
# 

# In[2]:


# to save filepath to variable 
melbourne_file_path = (r'C:\Users\User\Downloads\melb_data.csv\melb_data.csv')
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a detailed information  of the data in Melbourne data
melbourne_data.describe()


# In[3]:


melbourne_data.columns


# In[4]:


melbourne_data.isnull().sum()


# In[5]:


# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
#to have a good prediction dropping null alues is essential one
# dropna drops missing values 
melbourne_data = melbourne_data.dropna(axis=0)


# In[6]:


melbourne_data.isnull().sum()


# # Selecting the Prediction
# 
# here i want some features based upon the price value. so my prediction target is "Price"  
# by placing this as prediction i am extracting features like  ('Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude)]
# 

# #  Feature selection

# In[11]:


y = melbourne_data.Price


# In[7]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


# In[8]:


X = melbourne_data[melbourne_features]


# In[12]:


X.describe()


# ##Decision Tree — A tree algorithm used in machine learning to find patterns in data by learning decision rules.

# In[14]:


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)


# We now have a fitted model that we can use to make predictions.
# This helps to make predictions for new houses coming on the market rather than the houses we already have prices for. 
# But we'll make predictions for the first few rows of the training data to see how the predict function works

# In[15]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# # Model Validation

# If we suppose to  make predictions with our training data and compare those predictions to the target values in the training data definitely we see many problems and the data points will  be huge and it leads to high bias.
# 
# To overcome this we use some  metrics  called Mean Absolute Error (also called MAE). 
# 
# ***Mean Absolute Error ((error=actual−predicted))
# 
# 
# **we are importing mean absolute error from scikit learn**

# In[17]:


from sklearn.metrics import mean_absolute_error


# In[41]:


predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# #  Train and Test split
# 
# ** split data into training and validation data, for both features (x) and target(y).

# In[42]:


from sklearn.model_selection import train_test_split

# The split is based on a random number generator. 
#Supplying a numeric value to the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# so mean absolute error for the in-sample data was about 1100 dollars. Out-of-sample it is more than 260,000 dollars.

# ## Underfitting and Overfitting

# When we divide the houses amongst many leaves, we also have fewer houses in each leaf. Leaves with very few houses will make predictions that are quite close to those homes' actual values, but they may make very unreliable predictions for new data (because each prediction is based on only a few houses).
# 
# **This is a phenomenon called overfitting.
# 
# At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses. Resulting predictions may be far off for most houses, even in the training data (and it will be bad in validation too for the same reason). When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data.
# 
# **This is a phenomenon called underfitting.
# 

# In[45]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[46]:


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 555]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))


# In this we see a overfitting and underfitting problems with respect to the leaf nodes.. 
# for overcoming such problems we use random forest method.

# # Random Forest

# ###  Random forests are an ensemble of decision trees that can improve the accuracy of decision trees by reducing overfitting

# In[47]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
m_preds = forest_model.predict(val_X)  #where m_preds= melbourne predictions
print(mean_absolute_error(val_y, m_preds))


# # CONCLUSION

# ** here in Decision trees we get multiple mean absolute errors with the quantity of leaf nodes...
# 
# but in Random Forest in ensemble with multiple decision trees which improve accuracy and giving more accurate value of mean absolute error.
