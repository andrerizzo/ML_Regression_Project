# -*- coding: utf-8 -*-
"""
Script for Model Training and Testing
Author: Andre Rizzo
Version: 1.1
"""

import pandas as pd
import numpy as np

# Create Train and Test Sets
from sklearn.model_selection import train_test_split

# Categorical Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Import function
import Regression_Model_Train_and_Test


# Load dataset
df = pd.read_csv("/Users/andre.rizzo/OneDrive/Personal Projects/Carprice/data/raw/audi.csv")

# Create a copy of the original dataset
df_mod = df

# Create X and y datasets
X = df_mod.drop(['model', 'price'], axis=1)
y = df_mod.loc[:, 'price'].values
y = np.array(y).reshape(len(y), 1)

# Encode categorical data
cat_transform = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='first'), ['transmission', 'fuelType'])],
    remainder='passthrough')
X = cat_transform.fit_transform(X)

# Split train ant test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


##################################################################################################
#               Model Training Using Hyperparameters Previously Optimized
##################################################################################################
#Regression_Model_Train_and_Test.ridge_regression_model_train_and_test(X_train, X_test, y_train, y_test,
#                                                                      alpha=99.96426216953114,
#                                                                      solver='sag')
# Regression_Model_Train_and_Test.lasso_regression_model_train_and_test(X_train, X_test, y_train, y_test,
#                                                                       alpha=0.32506244798120437)
Regression_Model_Train_and_Test.knn_regression_model_train_and_test(X_train, X_test, y_train, y_test,
                                                                     n_neighbors=11,
                                                                     weights='uniform',
                                                                     p=1,
                                                                     algorithm='ball_tree')
# Regression_Model_Train_and_Test.gradient_boosting_regression_model_train_and_test(X_train, X_test, y_train, y_test,
#                                                                                   n_estimators=80,
#                                                                                   loss='absolute_error',
#                                                                                   max_depth=100,
#                                                                                   max_leaf_nodes=85)
# Regression_Model_Train_and_Test.xgboost_regression_model_train_and_test(X_train, X_test, y_train, y_test,
#                                                                         gamma=0.41,
#                                                                         max_depth=44,
#                                                                         min_child_weight=11.50,
#                                                                         learning_rate=0.43,
#                                                                         reg_lambda=0.88,
#                                                                         reg_alpha=0.14,
#                                                                         subsample=0.835)
