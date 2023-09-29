import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report
import pickle
heart = pd.read_csv(r"veriseti.csv")
heart['SOY'].value_counts()
heart['SOY'].value_counts()/heart.shape[0]*100
labels=['Yes','No']
values=heart['SOY'].value_counts().values
target = heart['SOY']
features = heart.drop(['SOY'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.12, random_state = 0)
def fit_eval_model(model, train_features, y_train, test_features, y_test):
    results = {}
    model.fit(train_features, y_train)
    test_predicted = model.predict(test_features)
    results['classification_report'] = classification_report(y_test, test_predicted)
    results['confusion_matrix'] = confusion_matrix(y_test, test_predicted)
    return results
sv = SVC(random_state = 1)
rf = RandomForestClassifier(random_state = 1)
ab = AdaBoostClassifier(random_state = 1)
gb = GradientBoostingClassifier(random_state = 1)
xg = XGBClassifier(random_state = 1) 
results = {}
models = [sv, rf, ab, gb, xg] 
for cls in models:
    cls_name = cls.__class__.__name__
    results[cls_name] = {}
    results[cls_name] = fit_eval_model(cls, X_train, y_train, X_test, y_test)
best_model = None
best_score = 0
for result in results:
    score = results[result]['classification_report'].split('\n')[-2].split()[-2] 
    score = float(score)
    if score > best_score:
        best_score = score
        best_model = result
for result in results:
    print (result)
    print()
    for i in results[result]:
        print (i, ':')
        print(results[result][i])
        print()
    print ('-----')
    print()
importance = None
if best_model == 'XGBClassifier': 
    importance = xg.feature_importances_
elif best_model == 'GradientBoostingClassifier':
    importance = gb.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (features.columns[i], v))
feature_names = features.columns 
plt.barh(feature_names, importance)
plt.ylabel('Özellikler')
plt.xlabel('Önem Skorları')
plt.title('Özelliklerin Önem Skorları')
plt.yticks(rotation=0)
plt.show()
if best_model == 'XGBClassifier': 
    with open('XGBClassifier.pkl', 'wb') as file:
        pickle.dump(xg, file)
elif best_model == 'GradientBoostingClassifier':
    with open('GradientBoostingClassifier.pkl', 'wb') as file:
        pickle.dump(gb, file)
elif best_model == 'AdaBoostClassifier': 
    with open('AdaBoostClassifier.pkl', 'wb') as file:
        pickle.dump(ab, file)
elif best_model == 'RandomForestClassifier': 
    with open('RandomForestClassifier.pkl', 'wb') as file:
        pickle.dump(rf, file)
elif best_model == 'SVC': 
    with open('SVC.pkl', 'wb') as file:
        pickle.dump(sv, file)
print("Best model: ", best_model)
print("F1-score: ", best_score)