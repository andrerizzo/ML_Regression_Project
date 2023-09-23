# -*- coding: utf-8 -*-
"""
Script to perform Regression
Author: Andre Rizzo
Version: 1.1
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create Train and Test Sets
from sklearn.model_selection import train_test_split

# Categorical Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import Regression_Model_Evaluation


# Load dataset
df = pd.read_csv("../../data/raw/audi.csv")

# Create a copy of the original dataset
df_mod = df

# Create X and y datasets
X = df_mod.drop(['model', 'price'], axis=1)
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
# Check which models will be a final candidate
##################################################################################################

result = {
    "Linear Regression": Regression_Model_Evaluation.linear_regression_model_evaluation(X_train, y_train),
    "Ridge Regression": Regression_Model_Evaluation.ridge_regression_model_evaluation(X_train, y_train),
    "Lasso Regression": Regression_Model_Evaluation.lasso_regression_model_evaluation(X_train, y_train),
    "KNN Regression": Regression_Model_Evaluation.knn_regression_model_evaluation(X_train, y_train),
    "Support Vector Regression": Regression_Model_Evaluation.support_vector_regression_model_evaluation(X_train, y_train),
    "Elastic Net Regression": Regression_Model_Evaluation.elastic_net_regression_model_evaluation(X_train, y_train),
    "Random Forest Regression": Regression_Model_Evaluation.random_forest_regression_model_evaluation(X_train, y_train),
    "Decision Tree Regression": Regression_Model_Evaluation.decision_tree_regression_model_evaluation(X_train, y_train),
    "AdaBoost Regression": Regression_Model_Evaluation.adaboost_regression_model_evaluation(X_train, y_train),
    "Multi Layer Perceptron Regression": Regression_Model_Evaluation.neural_network_regression_model_evaluation(X_train, y_train),
    "Gradient Boosting Regression": Regression_Model_Evaluation.gradient_boosting_regression_model_evaluation(X_train, y_train),
    "XGBoost Regression": Regression_Model_Evaluation.xgboost_regression_model_evaluation(X_train, y_train)
}


# Sort results by MSE and print
result = sorted(result.items(), reverse=True, key=lambda item: item[1])
print("")
print("#################################################")
print("#            Model Candidates Result            #")
print("#################################################")

for key, value in result:
    print(key, ":", value)

selection_model = open("../../reports/model_performance_full.txt", "w+")
selection_model.write(str(result))
selection_model.close()












