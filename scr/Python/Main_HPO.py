# -*- coding: utf-8 -*-
"""
Script for Hyperparameter Tuning Using Bayesian Optimization
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
import HPO_with_Random_Search
import HPO_with_Bayesian_Optimization


# Load dataset
df = pd.read_csv("/Users/andre.rizzo/OneDrive/Personal Projects/Carprice/data/raw/audi.csv")

# Create a copy of the original dataset
df_mod = df

# Create X and y datasets
X = df_mod.drop(['model','price'], axis=1)
y = df_mod.loc[:,'price'].values
y = np.array(y).reshape(len(y), 1)

# Encode categorical data
cat_transform = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='first'), ['transmission', 'fuelType'])],
    remainder='passthrough')
X = cat_transform.fit_transform(X)

# Split train ant test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


##################################################################################################
#               Hyperparameter Optimization Using Random Search
##################################################################################################
# HPO_with_Random_Search.ridge_regression_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.lasso_regression_model_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.knn_regression_hpo_rs(X_train, y_train)
HPO_with_Random_Search.support_vector_regression_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.elastic_net_regression_model_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.random_forest_regression_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.decision_tree_regression_model_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.adaboost_regression_model_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.neural_network_regression_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.gradient_boosting_regression_hpo_rs(X_train, y_train)
# HPO_with_Random_Search.xgboost_regression_model_hpo_rs(X_train, y_train)


##################################################################################################
#               Hyperparameter Optimization Using Bayesian Optimization
##################################################################################################
# HPO_with_Bayesian_Optimization.ridge_regression_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.lasso_regression_model_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.knn_regression_hpo_bo(X_train, y_train)
HPO_with_Bayesian_Optimization.support_vector_regression_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.elastic_net_regression_model_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.random_forest_regression_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.decision_tree_regression_model_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.adaboost_regression_model_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.neural_network_regression_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.gradient_boosting_regression_hpo_bo(X_train, y_train)
# HPO_with_Bayesian_Optimization.xgboost_regression_model_hpo_bo(X_train, y_train)
