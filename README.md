# kaggle_credit 
Codes for Kaggle competition "Give me some credit"

This repository contains codes and notebooks for the Kaggle competition titled "Give me some credit". [1]
This competition is expected to use some financial data of people and predict if they are likely to face financial issues or not.
The task is set to use AUC as it is an informative method to evaluate binary classification.

# Questions to answer
Below we will answer some questions regarding methodology and metrics
## Why AUC-ROC?

Area Under the Curve of the Receiver Operator Characteristic shows how well our binary classifier can distinguish between our two classes.
This comes handy as we want to know what is the True Positive Rate (TPR) and False Positive Rate (FPR) of our model and this one figure will give us the answer to both.
An important reason for using AUC-ROC is that since our data is imbalanced the model can get a high precision only with predicting the whole validation data as 0. Which we know is not a correct behavior.  

Other potential metric to use:

1) Logloss: To penalize our model on the prediction when it is certain our some predictions.
This loss is implemented in sklearn as: sklearn.metrics.log_loss and can be calculated as:
`log_loss = -(y * log(y_hat) + (1-y) * (1-y_hat))`. Where y is true label and y_hat is prediction from the model

2) MSE: Mean Squared Error or in short MSE is another metric to evaluate our regression model.
It is basically how far our prediction is from the true value.
`MSE = (y_hat - y)**2`
MSE can be used alongside MSE for a better understanding of model's performance.

Others: Cohen-Kappa score is a measure to say how much two annotators agree on an annotation. It can be used a measure for our model. F1 score, harmonic mean between precision and recall. 

## How to validate the model?
In order to validate the model I have used K-fold cross-validation as it a well-known robust method to the model validation.
In this method we will do the following:
 1) Shuffle the dataset
 2) split the dataset into k different folds
 3) for each fold:
    1) Take the group as validation set
    2) Train the model on the other group
    3) predict the current group and calculate the loss/evaluation metric
 4) Summarize (Maybe average) over the whole k folds.
 
 I have used this method because it 1) shuffles the data so that we won't end up with easy validation set (maybe the last 20% of the dataset is easy). 2) Uses all the training dataset as evaluation so this will give us a better understanding of model generalization.
 
# Introduction
There are multiple notebooks in the notebooks directory that contain the different approaches that I have taken.


# Data Analysis
In order to analyse the data I have used pandas and some simple statistics to understand how the data is distributed.

## Labels balance
> related notebook: `notebooks/data_exploration.ipynb`

In order to check if we have a balanced or unbalanced dataset. I have grouped the datapoints by the count of labels (SeriousDlqin2yrs).
Turned out that we have the following data:
```
SeriousDlqin2yrs
0    139974
1     10026

```
This is something that I can later look at. Maybe using a balanced bagging classifier could be a good idea.

## Explaining important features using RandomForest
>Related notebook: `notebooks/feature_selection_rf.ipynb`

In order to find out the top features we use RandomForest as it is an explainable model and look at the top features it uses to predict the datapoints:

```python
{'RevolvingUtilizationOfUnsecuredLines': 0.19221123680189298,
 'age': 0.12171943243752772,
 'NumberOfTime30-59DaysPastDueNotWorse': 0.03604861933861011,
 'DebtRatio': 0.19444260090080842,
 'MonthlyIncome': 0.14865628566881625,
 'NumberOfOpenCreditLinesAndLoans': 0.08066758193948617,
 'NumberOfTimes90DaysLate': 0.12786316249439175,
 'NumberRealEstateLoansOrLines': 0.03018097017039138,
 'NumberOfTime60-89DaysPastDueNotWorse': 0.02865017488715702,
 'NumberOfDependents': 0.0395599353609183}
```
> Note: As Sklearn documentations has suggested: impurity-based feature importances can be misleading for high cardinality features (many unique values)

Based on the results above the top features are:

```
 'age': 0.12171943243752772,
 'NumberOfTimes90DaysLate': 0.12786316249439175,
 'MonthlyIncome': 0.14865628566881625,
 'RevolvingUtilizationOfUnsecuredLines': 0.19221123680189298,
 'DebtRatio': 0.19444260090080842
```
 
## Training models only on top features
> Related notebook: `notebooks/feature_selection_rf`

Now we want to train a model only on the important features. Based on my previous experience this method, sometimes, yields
very good results. After training RF and Gradient Boosting models. I came to the conclusion that this method of feature selection is not suitable.

## Hyperparameter tuning for Gradient Boosting
> Related notebook: `notebooks/gb_gridsearch`

For this section, I tried to use all the features with our Gradient Boosting model and try to use Gridsearch to find the best parameters for the model.




# Final Model 

# How to run?

## Docker

## Python

# Requirements

# Ideas
1) BaggingClassifier

# References
[1] https://www.kaggle.com/c/GiveMeSomeCredit/