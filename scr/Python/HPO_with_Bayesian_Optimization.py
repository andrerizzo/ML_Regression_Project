# -*- coding: utf-8 -*-
"""
Script to perform Hyperparameters Optimization Using Bayesian Optimization
Author: Andre Rizzo
Version: 2.2
"""


from time import time
from datetime import datetime
import numpy as np



def ridge_regression_hpo_bo(X_train, y_train):

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Ridge Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = Ridge(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Define Cross Validation Method
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # Hyperparameters
    hyperparameter = {
        # 'alpha': hp.uniform('alpha', 1e-5, 100),
        'alpha': hp.loguniform('alpha', 0, 2),
        # 'fit_intercept' : hp.choice('fit_intercept', [True, False]),
        'fit_intercept': hp.choice('fit_intercept', [False]),
        # 'tol': hp.loguniform('tol', 1e-5, 1),
        # 'solver': hp.choice('solver', ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
        'solver': hp.choice('solver', ['svd'])
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Ridge Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#          HPO for Ridge Regression Model              #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Ridge_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Ridge Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def lasso_regression_model_hpo_bo(X_train, y_train):

    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Lasso Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = Lasso(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

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

    # Hyperparameters
    hyperparameter = {
        # 'alpha': hp.loguniform('alpha', 1e-5, 100),
        # 'fit_intercept': hp.choice('fit_intecept', [True, False]),
        # 'tol': hp.loguniform('tol', 1e-5, 1),
        # 'selection': hp.choice('selection', ['cyclic', 'random'])
        'alpha': hp.loguniform('alpha', 1e-5, 1),
        'fit_intercept': hp.choice('fit_intercept', [True]),
        'tol': hp.loguniform('tol', 1e-5, 1),
        'selection': hp.choice('selection', ['random'])
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Lasso Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#           HPO for Lasso Regression Model             #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Lasso_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Lasso Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def knn_regression_hpo_bo(X_train, y_train):

    # Import libraries
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from hyperopt.pyll import scope
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for KNN Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = KNeighborsRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)
    X_train = nzv.fit_transform(X_train)

    # Using Validation Set with 10-fold Cross Validation
    cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = {
        'n_neighbors': scope.int(hp.quniform('n_neighbors',1, 20, 1)),
        'weights': hp.choice('weigths',['uniform', 'distance']),
        'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
        'p': scope.int(hp.quniform('p',1, 3, 1))
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for KNN Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#           HPO for KNN Regression Model               #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/KNN_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("KNN Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def support_vector_regression_hpo_bo(X_train, y_train):

    # Import libraries
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from hyperopt.pyll import scope
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for SVM Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = SVR(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    y_train = y_train.ravel()

    # Modeling using kNN Regression
    svm_fit = SVR()

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = {
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': scope.int(hp.quniform('degree', 1, 20, 1)),
        'gamma': hp.choice('gamma', ['scale', 'auto']),
        'coef0': hp.loguniform('coef0', 1e-5, 100),
        'C': hp.loguniform('C', 1e-5, 100),
        'epsilon': hp.uniform('epsilon', 1e-5, 100)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for SVM Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#           HPO for SVM Regression Model               #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/SVM_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Support Vector Machine Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def elastic_net_regression_model_hpo_bo(X_train, y_train):

    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Elastic Net Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = ElasticNet(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

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

    # HPO using Random Search
    hyperparameter = {
        'alpha': hp.uniform('alpha', 1e-5, 100),
        'l1_ratio': hp.uniform('l1_ratio', 0, 1)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Elastic Net Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#       HPO for Elastic Net Regression Model           #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Elastic_Net_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Elastic Net Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def random_forest_regression_hpo_bo(X_train, y_train):

    # Import libraries
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Random Forest Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = RandomForestRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # Hyperparameters
    hyperparameter = {
        'n_estimators': hp.uniformint('n_estimators', 1, 200),
        'max_depth': hp.uniformint('max_depth', 1, 200),
        'criterion': hp.choice('criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']),
        'min_samples_split': hp.uniformint('min_samples_split', 1, 100),
        'min_samples_leaf': hp.uniformint('min_samples_split', 1, 100),
        'max_features': hp.uniform('max_features', 0.1, 2)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Random Forest Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#       HPO for Random Forest Regression Model         #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Random_Forest_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Random Forest Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def decision_tree_regression_model_hpo_bo(X_train, y_train):

    # Import libraries
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Decision Tree Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = DecisionTreeRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=2)

    # HPO using Random Search
    hyperparameter = {
        'criterion': hp.choice('criterion', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        'max depth': hp.uniformint('max_depth', 1, 100),
        'min samples split': hp.uniformint('min samples split', 1, 100),
        'min samples leaf': hp.uniformint('min samples split', 1, 100),
        'max features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
        'min_weight_fraction_leaf': hp.lognormal('min_weight_fraction_leaf', 1e-5, 100),
        'max_leaf_nodes': hp.uniformint('max_leaf_nodes', 1, 100)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Ridge Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#       HPO for Decision Tree Regression Model         #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Decision_Tree_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Decision Tree Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def adaboost_regression_model_hpo_bo(X_train, y_train):

    # Import libraries
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Ada Boost Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = AdaBoostRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Random Search
    hyperparameter = {
        'n estimators': hp.uniformint('n estimators', 1, 100),
        'learning rate': hp.uniform('learning rate', 0, 100)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Adaboost Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#         HPO for AdaBoost Regression Model            #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/AdaBoost_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("AdaBoost Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def neural_network_regression_hpo_bo(X_train, y_train):

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Logistic Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = MLPRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    y_train = y_train.ravel()

    # Remove Variables with Zero Variance
    nzv = VarianceThreshold(threshold=0)

    # Modeling using Neural Networks Regression
    neural_fit = MLPRegressor(random_state=1)

    # Define model evaluation method
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=1)

    # HPO using Random Search
    hyperparameter = {
        'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
        'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
        'alpha': hp.uniform('alpha', 1e-5, 100),
        'batch_size': 'auto',
        'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
        'learning_rate_init': hp.uniform('learning_rate_init', 1e-5, 100),
        # 'power_t': hp.uniform('power_t', 1e-5, 100),
        'max_iter': hp.uniformint('max_iter', 50, 500)
        # 'momentum': hp.uniform('momentum', 0, 1),
        # 'beta_1': hp.uniform('beta_1', 0, 0.99999),
        # 'beta_2': hp.uniform('beta_2', 0, 0.99999)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for MLP Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#          HPO for MLP Regression Model                #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/MLP_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Multi Layer Perceptron Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def gradient_boosting_regression_hpo_bo(X_train, y_train):

    # Import libraries
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for Gradient Boosting Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = GradientBoostingRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)
    y_train = y_train.ravel()

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Bayesian Optimization
    hyperparameter = {
        'n_estimators': hp.uniformint('n_estimators', 1, 100),
        'loss': hp.choice('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
        'max_depth': hp.uniformint('max_depth', 1, 100),
        'max_leaf_nodes': hp.uniformint('max_leaf_nodes', 1, 100)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for Gradient Boosting Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#     HPO for Gradient Boosting Regression Model       #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/Gradient_Boosting_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("Gradient Boosting Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()


def xgboost_regression_model_hpo_bo(X_train, y_train):

    # Import libraries
    from xgboost import XGBRegressor
    from sklearn.model_selection import RepeatedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from hyperopt import tpe
    from hyperopt import STATUS_OK
    from hyperopt import Trials
    from hyperopt import hp
    from hyperopt import fmin
    from math import sqrt

    # Initialize variables
    num_folds = 5
    max_evaluation = 50

    def objective(params, n_folds=num_folds):
        # Objective function for XGBoost Regression Hyperparameter Tuning

        # Perform n_fold cross validation with hyperparameters
        model_fit = XGBRegressor(**params)
        scores = cross_val_score(model_fit, X_train, y_train, cv=n_folds, scoring='neg_root_mean_squared_error')

        # Extract the best score and loss must be minimized
        loss = scores.mean()

        # Return all relevant information
        return {'loss': -loss, 'params': params, 'status': STATUS_OK}

    # Center and Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # Using Validation Set with k-fold Cross Validation
    # cv = RepeatedKFold(n_splits=10, n_repeats=2, random_state=2)

    # HPO using Bayesian Optimization
    hyperparameter = {
        #'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        #'verbosity': 2,
        #'validate_parameters': True,
        'booster': 'gblinear',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'reg_lambda': hp.loguniform('reg_lambda', 1e-6, 2.),
        'reg_alpha': hp.loguniform('reg_alpha', 1e-6, 2.),
        'updater': hp.choice('updater', ['shotgun', 'coord_descent']),
        'feature_selector': hp.choice('feature_selector', ['cyclic', 'shuffle'])
        #'learning_rate': hp.loguniform('learning_rate', 1e-3, 1.),
        #'gamma': hp.uniform('gamma', 0, 100),
        #'max_depth': hp.uniformint('max_depth', 0, 100),
        #'min_child_weight': hp.uniformint('min_child_weight',1, 10),
        #'subsample': hp.uniform('subsample', 0, 0.9)
    }

    # Look for the best hyperparameters
    print("")
    print("Performing HPO for XGBoost Regression Model")
    start = time()
    tpe_algorithm = tpe.suggest
    bayes_trials = Trials()
    seed = np.random.default_rng(1)
    best = fmin(fn=objective, space=hyperparameter, algo=tpe_algorithm, max_evals=max_evaluation, trials=bayes_trials,
                rstate=seed)
    finish = time()

    # Check model performance
    print("")
    print("########################################################")
    print("#         HPO for XGBoost Regression Model             #")
    print("########################################################")
    res = bayes_trials.best_trial.get('result')
    rmse = sqrt(res.pop('loss'))
    print('Hyperparameters: ', res.get('params'))
    print('RMSE = ', rmse)
    print('Time elapsed: %f seconds' % (finish - start))
    print('\n')

    # Save params to a text file for future use
    current_datetime = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    file_name = "../../reports/XGBoost_Regression_Hyperparams_BO_" + current_datetime + ".txt"
    hyper_file = open(file_name, "w+")
    hyper_file.write("XGBoost Regression")
    hyper_file.write('\n')
    hyper_file.write('\n')
    hyper_file.write("Hyperparameters:")
    hyper_file.write(str(res.get('params')))
    hyper_file.write('\n')
    hyper_file.write("RMSE = ")
    hyper_file.write(str(rmse))
    hyper_file.close()

