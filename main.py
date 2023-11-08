# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Load the dataset
heart = pd.read_excel("dataset.xlsx")

# Check for missing data (optional)
print(heart.isnull().sum())

# Prepare the data (handling missing data, encoding categorical data, scaling numerical data, etc.)
# Data preprocessing is optional.

# Split the dataset into features and target
target = heart['SOY']
features = heart.drop(['SOY'], axis=1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.12, random_state=0)

# Define a function to evaluate model performance
def evaluate_model(model, train_features, y_train, test_features, y_test):
    results = {}
    model.fit(train_features, y_train)
    test_predicted = model.predict(test_features)
    results['classification_report'] = classification_report(y_test, test_predicted)
    results['confusion_matrix'] = confusion_matrix(y_test, test_predicted)
    return results

# Choose the models to evaluate
models = [SVC(random_state=1), RandomForestClassifier(random_state=1), AdaBoostClassifier(random_state=1),
          GradientBoostingClassifier(random_state=1), XGBClassifier(random_state=1)]

best_model = None
best_score = 0

results = {}

# Train and evaluate models in a loop
for model in models:
    model_name = model.__class__.__name__
    results[model_name] = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Find the model with the best F1 score
    f1_score = float(results[model_name]['classification_report'].split('\n')[-2].split()[-2])

    if f1_score > best_score:
        best_score = f1_score
        best_model = model_name

    print(model_name)
    print()
    for key, value in results[model_name].items():
        print(key, ':')
        print(value)
        print()

    print('-----')
    print()

# Print the best model and its F1 score
print("Best model: ", best_model)
print("F1 score: ", best_score)

# Save the best model (using pickle)
best_model_instance = None

if best_model == 'XGBClassifier':
    best_model_instance = XGBClassifier(random_state=1)
elif best_model == 'GradientBoostingClassifier':
    best_model_instance = GradientBoostingClassifier(random_state=1)
elif best_model == 'AdaBoostClassifier':
    best_model_instance = AdaBoostClassifier(random_state=1)
elif best_model == 'RandomForestClassifier':
    best_model_instance = RandomForestClassifier(random_state=1)
elif best_model == 'SVC':
    best_model_instance = SVC(random_state=1)

best_model_instance.fit(X_train, y_train)

with open(f'{best_model}.pkl', 'wb') as file:
    pickle.dump(best_model_instance, file)

# Plot feature importance scores (for XGBoost and Gradient Boosting models)
importance = None

if best_model == 'XGBClassifier':
    importance = best_model_instance.feature_importances_
elif best_model == 'GradientBoostingClassifier':
    importance = best_model_instance.feature_importances_

if importance is not None:
    feature_names = features.columns
    plt.barh(feature_names, importance)
    plt.ylabel('Features')
    plt.xlabel('Importance Scores')
    plt.title('Feature Importance Scores')
    plt.yticks(rotation=0)
    plt.show()
