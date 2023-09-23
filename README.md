# ML_Regression_Project

### Objective:
Perform Machine Learning regression techniques to predict car prices.

### Details
The project is structured using two distinct sections.  
</br>

**Section 1**  
This section was completely written in **R Language** and is responsible to perform the following tasks:  
* Download data from Kaggle.  
* Perform Exploratory Data Analysis.
* Perform Feature Engineering.
* Perform Feature Selection.
</br>

**Section 2**  
This section was written in **Python** and is responsible to perform the following tasks:  
* Split data into train and testing sets.  
* Perform an initial modeling using twelve ML algorithms. Here, the main objective is to select the best five
  algorithms that will have the hyperparameters tuned.  
* The regression algorithms used above were **Linear Regression**, **Ridge Regression**, **Lasso**, **KNN**,
  **Support Vector Machine**, **Elastic Net**, **Random Forests**, **Decision Trees**, **AdaBoost**,
  **Multi Layer Perceptron**, **Gradient Boosting**, **XGBoost**.
* For the five models with the smallest RMSEs I performed a first step of hyperparameter optimization using
  **Random Search**. This method was choosen because it is not so time consuming as **Grid Search**.  
* With a better idea of the hyperparameters to be used, I performed a second step of tuning, using
  **Bayesian Optimization**.
* All the hyperparameters optimization steps were done using the **cross-validation method** performed on
  the training set. This action was very important because test set can only be used in the last phase just
  to check the final model performance.
* With the hyperparameters defined, the final modeling was performed.  
