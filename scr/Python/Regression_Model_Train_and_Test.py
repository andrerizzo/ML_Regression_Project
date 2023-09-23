# -*- coding: utf-8 -*-
"""
Script to perform Regression Model's Training and Test
Author: Andre Rizzo
Version: 1.1
"""


def ridge_regression_model_train_and_test(X_train, X_test, y_train, y_test, alpha, solver):

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Modeling using Ridge Regression
    model = Ridge(alpha=alpha, solver=solver, random_state=1)
    model.fit(X_train, y_train)

    # Predict using Test Set
    y_hat = model.predict(X_test)

    # Check model performance
    print("")
    print("########################################################")
    print("#               Ridge Regression Model                 #")
    print("########################################################")
    print(model)
    print('RMSE= ', np.sqrt(mean_squared_error(y_test, y_hat)))
    print('R2= ', r2_score(y_test, y_hat), "\n")


def lasso_regression_model_train_and_test(X_train, X_test, y_train, y_test, alpha):

    from sklearn.linear_model import Lasso
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    import numpy as np

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Modeling using Lasso
    model_fit = Lasso(alpha=alpha)
    model = model_fit.fit(X_train, y_train)

    # Predict using Test Set
    y_pred = model.predict(X_test)

    # Check model performance
    print("")
    print("########################################################")
    print("#               Lasso Regression Model                 #")
    print("########################################################")
    print(model)
    rmse = metrics.mean_squared_error(y_test, y_pred)
    print('RMSE= ', np.sqrt(rmse))
    print('R2= ', metrics.r2_score(y_test, y_pred))
    print("")


def knn_regression_model_train_and_test(X_train, X_test, y_train, y_test, n_neighbors, weights, p, algorithm):

    # Import libraries
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.feature_selection import VarianceThreshold
    from sklearn import metrics
    import numpy as np

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    X_test = scaler_X.fit_transform(X_train)
    y_test = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)
    X_test = nzv.fit_transform(X_test)

    # Modeling using KNN Regression
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p, algorithm=algorithm)
    model.fit(X_train, y_train)

    # Predict using Test Set
    y_hat = model.predict(X_test)

    # Check model performance
    print("")
    print("########################################################")
    print("#               KNN Regression Model                   #")
    print("########################################################")
    print(model)
    rmse = metrics.mean_squared_error(y_test, y_hat)
    print('RMSE= ', np.sqrt(rmse))
    print('R2= ', metrics.r2_score(y_test, y_hat))
    print("")


def gradient_boosting_regression_model_train_and_test(X_train, X_test, y_train, y_test, n_estimators,
                                                      loss, max_depth, max_leaf_nodes):

    # Import libraries
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    y_train = y_train.ravel()

    # Modeling using Gradient Boosting Regression
    model = GradientBoostingRegressor(n_estimators=n_estimators, loss=loss, max_depth=max_depth,
                                      max_leaf_nodes=max_leaf_nodes)
    model.fit(X_train, y_train)

    # Predict using Test Set
    y_pred = model.predict(X_test)

    # Check model performance
    print("")
    print("########################################################")
    print("#          Gradient Boosting Regression Model          #")
    print("########################################################")
    print(model)
    rmse = metrics.mean_squared_error(y_test, y_pred)
    print('RMSE= ', np.sqrt(rmse))
    print('R2= ', metrics.r2_score(y_test, y_pred))
    print("")


def xgboost_regression_model_train_and_test(X_train, X_test, y_train, y_test, gamma, max_depth, min_child_weight,
                                            learning_rate, reg_lambda, reg_alpha, subsample):

    # Import libraries
    import numpy as np
    from xgboost import XGBRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Modeling using XGBoost Regression
    model = XGBRegressor(gamma=gamma, max_depth=max_depth, min_child_weight=min_child_weight,
                         learning_rate=learning_rate, reg_lambda=reg_lambda, reg_alpha=reg_alpha,
                         subsample=subsample)
    model.fit(X_train, y_train)

    # Predict using Test Set
    y_pred = model.predict(X_test)

    # Check model performance
    print("")
    print("########################################################")
    print("#             XGBoost Regression Model                 #")
    print("########################################################")
    print(model)
    rmse = metrics.mean_squared_error(y_test, y_pred)
    print('RMSE= ', np.sqrt(rmse))
    print('R2= ', metrics.r2_score(y_test, y_pred))
    print("")


