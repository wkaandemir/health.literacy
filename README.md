# Machine Learning
![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![scikit-learnn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)

The code begins by importing necessary Python libraries, loading a dataset, and preparing the data. It then splits the data into training and test sets. Next, it defines a function to evaluate the performance of different machine learning models, including Support Vector Classifier (SVC), Random Forest, AdaBoost, Gradient Boosting, and XGBoost. It iteratively trains and evaluates each of these models using the training and test data. The F1 score is used to select the best-performing model. The code saves the best model using the Pickle library and, if the best model is XGBoost or Gradient Boosting, it plots feature importance scores. Finally, it prints the name of the best model and its corresponding F1 score.

In summary, the code automates the process of model selection and evaluation for a classification problem, saving the best-performing model for future use, and optionally visualizing feature importance for specific models.

