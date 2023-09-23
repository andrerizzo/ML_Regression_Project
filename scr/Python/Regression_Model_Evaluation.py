# -*- coding: utf-8 -*-
"""
Script to perform Regression Model's Evaluation
Author: Andre Rizzo
Version: 1.1
"""

import numpy as np


def linear_regression_model_evaluation(X_train, y_train):

    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score
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
    lm_fit = linear_model.LinearRegression()
    lm_fit.fit(X_train, y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Evaluate model
    scores = 0
    scores = cross_val_score(lm_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("Linear Regression model parameters: ", lm_fit.get_params())
    print("Linear Regression Finished (1/12) \n")
    return result


def ridge_regression_model_evaluation(X_train, y_train):

    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score
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
    ridge_fit = linear_model.Ridge()
    ridge_fit.fit(X_train, y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Evaluate model
    scores = 0
    scores = cross_val_score(ridge_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("Ridge Regression model parameters: ", ridge_fit.get_params())
    print("Ridge Regression model finished (2/12) \n")
    return result


def lasso_regression_model_evaluation(X_train, y_train):

    from sklearn.linear_model import Lasso
    from sklearn.model_selection import cross_val_score
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
    lasso_fit = Lasso()
    lasso_fit.fit(X_train, y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Evaluate model
    scores = 0
    scores = cross_val_score(lasso_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("Lasso Regression model parameters: ", lasso_fit.get_params())
    print("Lasso Regression model finished (3/12) \n")
    return result


def knn_regression_model_evaluation(X_train, y_train):

    # Import libraries
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.feature_selection import VarianceThreshold

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Modeling using KNN Regression
    knn_fit = KNeighborsRegressor()
    knn_fit.fit(X_train, y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Evaluate model
    scores = 0
    scores = cross_val_score(knn_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("KNN Regression model parameters: ", knn_fit.get_params())
    print("KNN Regression model Finished (4/12) \n")
    return result


def support_vector_regression_model_evaluation(X_train, y_train):

    # Import libraries
    import numpy as np
    from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedKFold

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Modeling using kNN Regression
    svm_fit = svm.SVR()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # Evaluate model
    scores = 0
    scores = cross_val_score(svm_fit, X_train, y_train.ravel(), cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("SVM Regression model parameters: ", svm_fit.get_params())
    print("SVM Regression model Finished (5/12) \n")
    return result


def elastic_net_regression_model_evaluation(X_train, y_train):

    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score
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
    elastic_fit = linear_model.ElasticNet()
    elastic_fit.fit(X_train, y_train)

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Evaluate model
    scores = 0
    scores = cross_val_score(elastic_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')

    # print("Elastic Net Regression model MSE: %.3f (%.3f)" % (np.mean(scores), np.std(scores)), "\n")

    # Update dictionary from accuracy results
    # results = {}
    result = np.mean(scores)
    print("Elastic Net Regression model parameters: ", elastic_fit.get_params())
    print("Elastic Net Regression model finished (6/12) \n")
    return result


def random_forest_regression_model_evaluation(X_train, y_train):

    # Import libraries
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedKFold

    # Modeling using kNN Regression
    rfr_fit = RandomForestRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # Evaluate model
    scores = 0
    scores = cross_val_score(rfr_fit, X_train, y_train.ravel(), cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("Random Forest Regression model parameters: ", rfr_fit.get_params())
    print("Random Forest Regression model finished (7/12) \n")
    return result


def decision_tree_regression_model_evaluation(X_train, y_train):

    # Import libraries
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedKFold

    # Modeling using kNN Regression
    decision_fit = DecisionTreeRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # Evaluate model
    scores = 0
    scores = cross_val_score(decision_fit, X_train, y_train.ravel(), cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')

    # print("Decision Tree Regression model MSE: %.3f (%.3f)" % (np.mean(scores), np.std(scores)), "\n")

    # Update dictionary from accuracy results
    # results = {}
    # results['Decision Tree Regression'] = np.mean(scores)
    result = np.mean(scores)
    print("Decision Tree Regression model parameters: ", decision_fit.get_params())
    print("Decision Tree Regression model finished (8/12) \n")
    return result


def adaboost_regression_model_evaluation(X_train, y_train):

    # Import libraries
    import numpy as np
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import cross_val_score
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
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # Evaluate model
    scores = 0
    scores = cross_val_score(ada_fit, X_train, y_train.ravel(), cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')

    # print("AdaBoost Regression model MSE: %.3f (%.3f)" % (np.mean(scores), np.std(scores)), "\n")

    # Update dictionary from accuracy results
    # results = {}
    # results['AdaBoost Regression'] = np.mean(scores)
    result = np.mean(scores)
    print("ADABoost Regression model parameters: ", ada_fit.get_params())
    print("ADABoost Regression model finished (9/12) \n")
    return result


def neural_network_regression_model_evaluation(X_train, y_train):

    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
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

    # Modeling using Neural Networks Regression
    neural_fit = MLPRegressor(random_state=1)
    neural_fit.fit(X_train, y_train.ravel())

    # Define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Evaluate model
    scores = 0
    scores = cross_val_score(neural_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')

    # print("Neural Network Regression model MSE: %.3f (%.3f)" % (np.mean(scores), np.std(scores)), "\n")

    # Update dictionary from accuracy results
    # results = {}
    # results['Neural Network Regression'] = np.mean(scores)
    result = np.mean(scores)
    print("MLP Regression model parameters: ", neural_fit.get_params())
    print("MLP Regression model finished (10/12) \n")
    return result


def gradient_boosting_regression_model_evaluation(X_train, y_train):

    # Import libraries
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Modeling using Gradient Boosting Regression
    gbr_fit = GradientBoostingRegressor()

    # Using Validation Set with k-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # Evaluate model
    scores = 0
    scores = cross_val_score(gbr_fit, X_train, y_train.ravel(), cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("Gradient Boosting Regression model parameters: ", gbr_fit.get_params())
    print("Gradient Boosting Regression model finished (11/12) \n")
    return result


def xgboost_regression_model_evaluation(X_train, y_train):

    # Import libraries
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.model_selection import cross_val_score
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
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # Evaluate model
    scores = 0
    scores = cross_val_score(xgbr_fit, X_train, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    result = np.mean(scores)
    print("XGBoost Regression model parameters: ", xgbr_fit.get_params())
    print("XGBoost Regression model finished (12/12) \n")
    return result

