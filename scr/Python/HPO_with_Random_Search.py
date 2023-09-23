# -*- coding: utf-8 -*-
"""
Script to perform Hyperparameters Optimization Using Random Search
Author: Andre Rizzo
Version: 1.1
"""

import numpy as np
from time import time
from datetime import datetime
from math import sqrt


def ridge_regression_hpo_rs(X_train, y_train):

    from scipy.stats import loguniform
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import RandomizedSearchCV

    # Number of folds to Cross Validation
    cv = 5

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Select Regression Algorithm
    ridge_fit = Ridge()

    # Define Cross Validation Method
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    # HPO using Random Search
    hyperparameter = dict()
    hyperparameter['alpha'] = loguniform(0, 100)
    hyperparameter['fit_intercept'] = [True, False]
    hyperparameter['tol'] = loguniform(1e-5, 1)
    hyperparameter['solver'] = ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    # Define Random Search
    model_fit = RandomizedSearchCV(ridge_fit, hyperparameter, cv=cv, scoring='neg_mean_squared_error', random_state=0)

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Ridge Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#          HPO for Ridge Regression Model              #")
    print("########################################################")
    rmse = sqrt(abs(model_fit.best_score_))
    print('Best Score: %s' % rmse)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Ridge_Regression_Hyperparams_RS_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Ridge Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(model_fit.best_params_))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def lasso_regression_model_hpo_rs(X_train, y_train):

    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform

    # Number of folds to Cross Validation
    cv = 5

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Define model evaluation method
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Select Regression Algorithm
    lasso_fit = Lasso()

    # HPO using Random Search
    hyperparameter = dict()
    hyperparameter['alpha'] = loguniform(1e-5, 10)
    hyperparameter['fit_intercept'] = [True, False]
    hyperparameter['tol'] = loguniform(1e-5, 1)
    hyperparameter['selection'] = ['cyclic', 'random']

    # Define Random Search
    model_fit = RandomizedSearchCV(lasso_fit, hyperparameter, cv=cv, scoring='neg_mean_squared_error', random_state=0)

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Lasso Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#           HPO for Lasso Regression Model             #")
    print("########################################################")
    rmse = sqrt(abs(model_fit.best_score_))
    print('Best Score: %s' % rmse)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Lasso_Regression_Hyperparams_RS_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Lasso Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(model_fit.best_params_))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def knn_regression_hpo_rs(X_train, y_train):

    # Import libraries
    from scipy.stats import loguniform
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RandomizedSearchCV

    # Number of folds to Cross Validation
    cv = 5

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Modeling using kNN Regression
    knn_fit = KNeighborsRegressor()

    # Using Validation Set with 10-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()
    hyperparameter['n_neighbors'] = list(range(1, 20))
    hyperparameter['weights'] = ['uniform', 'distance']
    hyperparameter['algorithm'] = ['ball_tree', 'kd_tree', 'brute']
    hyperparameter['p'] = list(range(1, 3))

    # Define Random Search
    model_fit = RandomizedSearchCV(knn_fit, hyperparameter, cv=cv, scoring='neg_mean_squared_error', random_state=0)

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for KNN Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#           HPO for KNN Regression Model               #")
    print("########################################################")
    rmse = sqrt(abs(model_fit.best_score_))
    print('Best Score: %s' % rmse)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/KNN_Regression_Hyperparams_RS_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("KNN Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(model_fit.best_params_))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def support_vector_regression_hpo_rs(X_train, y_train):

    # Import libraries
    from scipy.stats import loguniform
    from sklearn import svm
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedKFold

    # Number of folds to Cross Validation
    cv = 5

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Modeling using kNN Regression
    svm_fit = svm.SVR()

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()

    hyperparameter['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    hyperparameter['degree'] = list(range(1, 20))
    hyperparameter['gamma'] = ['scale', 'auto']
    hyperparameter['coef0'] = loguniform(1e-5, 100)
    hyperparameter['C'] = loguniform(1e-5, 100)
    hyperparameter['epsilon'] = loguniform(1e-5, 100)

    # Define Random Search
    model_fit = RandomizedSearchCV(svm_fit, hyperparameter, cv=cv, scoring='neg_mean_squared_error', random_state=0)

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for SVM Regression Model")
    start = time()
    model_fit.fit(X_train, y_train.ravel())
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#           HPO for SVM Regression Model               #")
    print("########################################################")
    rmse = sqrt(abs(model_fit.best_score_))
    print('Best Score: %s' % rmse)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/SVM_Regression_Hyperparams_RS_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("SVM Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(model_fit.best_params_))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def elastic_net_regression_model_hpo_rs(X_train, y_train):

    from sklearn.linear_model import ElasticNet
    from scipy.stats import loguniform
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Modeling using Linear Regression
    elastic_fit = ElasticNet()

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # HPO using Random Search
    hyperparameter = dict()
    # hyperparameter['C'] = loguniform(1e-5, 100)
    # hyperparameter['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # hyperparameter['epsilon'] = loguniform(1e-5, 100)

    # Optional
    # hyperparameter['gamma'] = ['scale', 'auto']
    # hyperparameter['coef0'] = loguniform(1e-5, 100)
    # hyperparameter['degree'] = list(range(1, 20))

    # Define Random Search
    model_fit = RandomizedSearchCV(elastic_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Elastic Net Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#       HPO for Elastic Net Regression Model           #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("Elastic_Net_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()


def random_forest_regression_hpo_rs(X_train, y_train):

    # Import libraries
    from scipy.stats import loguniform
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import RandomizedSearchCV

    # Modeling using kNN Regression
    rfr_fit = RandomForestRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()
    # hyperparameter['C'] = loguniform(1e-5, 100)
    # hyperparameter['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # hyperparameter['epsilon'] = loguniform(1e-5, 100)

    # Optional
    # hyperparameter['gamma'] = ['scale', 'auto']
    # hyperparameter['coef0'] = loguniform(1e-5, 100)
    # hyperparameter['degree'] = list(range(1, 20))

    # Define Random Search
    model_fit = RandomizedSearchCV(rfr_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Random Forest Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#       HPO for Random Forest Regression Model         #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("Random_Forest_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()


def decision_tree_regression_model_hpo_rs(X_train, y_train):

    # Import libraries
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import RepeatedKFold

    # Modeling using Decision Tree Regression
    dtr_fit = DecisionTreeRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()
    # hyperparameter['C'] = loguniform(1e-5, 100)
    # hyperparameter['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # hyperparameter['epsilon'] = loguniform(1e-5, 100)

    # Optional
    # hyperparameter['gamma'] = ['scale', 'auto']
    # hyperparameter['coef0'] = loguniform(1e-5, 100)
    # hyperparameter['degree'] = list(range(1, 20))

    # Define Random Search
    model_fit = RandomizedSearchCV(dtr_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Decision Tree Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#       HPO for Decision Tree Regression Model         #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("Decision_Tree_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()


def adaboost_regression_model_hpo_rs(X_train, y_train):

    # Import libraries
    import numpy as np
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Modeling using kNN Regression
    ada_fit = AdaBoostRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()
    # hyperparameter['C'] = loguniform(1e-5, 100)
    # hyperparameter['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # hyperparameter['epsilon'] = loguniform(1e-5, 100)

    # Optional
    # hyperparameter['gamma'] = ['scale', 'auto']
    # hyperparameter['coef0'] = loguniform(1e-5, 100)
    # hyperparameter['degree'] = list(range(1, 20))

    # Define Random Search
    model_fit = RandomizedSearchCV(ada_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Decision Tree Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#         HPO for AdaBoost Regression Model            #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("AdaBoost_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()


def neural_network_regression_hpo_rs(X_train, y_train):

    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)

    # Modeling using Neural Networks Regression
    neural_fit = MLPRegressor(random_state=1)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # HPO using Random Search
    hyperparameter = dict()
    # hyperparameter['C'] = loguniform(1e-5, 100)
    # hyperparameter['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # hyperparameter['epsilon'] = loguniform(1e-5, 100)

    # Optional
    # hyperparameter['gamma'] = ['scale', 'auto']
    # hyperparameter['coef0'] = loguniform(1e-5, 100)
    # hyperparameter['degree'] = list(range(1, 20))

    # Define Random Search
    model_fit = RandomizedSearchCV(neural_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for MLP Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#    HPO for Multilayer Perceptron Regression Model    #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("MLP_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()


def gradient_boosting_regression_hpo_rs(X_train, y_train):

    # Import libraries
    from sklearn.ensemble import GradientBoostingRegressor
    from scipy.stats import loguniform
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Define Regression Algorithm
    gbr_fit = GradientBoostingRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()
    hyperparameter['random_state'] = [1]
    hyperparameter['n_estimators'] = list(range(1, 100))
    # hyperparameter['loss'] = ['squared_error', 'absolute_error', 'huber', 'quantile']
    hyperparameter['max_depth'] = list(range(1, 100))
    hyperparameter['max_leaf_nodes'] = list(range(1, 100))

    # Define Random Search
    model_fit = RandomizedSearchCV(gbr_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Gradient Boosting Regression Model")
    start = time()
    model_fit.fit(X_train, y_train.ravel())
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#     HPO for Gradient Boosting Regression Model       #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("Gradient_Boosting_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()


def xgboost_regression_model_hpo_rs(X_train, y_train):

    # Import libraries
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBRegressor
    from scipy.stats import loguniform
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Modeling using kNN Regression
    xgbr_fit = XGBRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = dict()
    hyperparameter['gamma'] = loguniform(1e-5, 100)
    hyperparameter['max_depth'] = list(range(0, 100))
    hyperparameter['min_child_weight'] = loguniform(1e-5, 100)
    hyperparameter['learning_rate'] = loguniform(1e-5, 1)
    # hyperparameter['booster'] = ['gbtree', 'gblinear', 'dart']
    hyperparameter['reg_lambda'] = loguniform(1e-5, 1)
    hyperparameter['reg_alpha'] = loguniform(1e-5, 1)
    hyperparameter['subsample'] = loguniform(1e-5, 1)
    hyperparameter['random_state'] = [44]

    # Define Random Search
    model_fit = RandomizedSearchCV(xgbr_fit, hyperparameter, cv=cv, scoring='r2')

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Gradient Boosting Regression Model")
    start = time()
    model_fit.fit(X_train, y_train)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#         HPO for XGBoost Regression Model             #")
    print("########################################################")
    print('Best Score: %s' % model_fit.best_score_)
    print('Best Hyperparameters: %s' % model_fit.best_params_)
    print('Time elapsed: %f seconds' % (finish - start))
    print("")

    with open("XGBoost_Hyperparameters.txt", 'w') as f:
        f.write(str(model_fit.best_params_))
        f.close()

