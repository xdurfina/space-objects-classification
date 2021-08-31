import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# Loading data
galaxy_train = pd.read_csv("train.csv")
galaxy_test = pd.read_csv("test.csv")

# Dropping NaN values
galaxy_train.dropna(how='any', inplace=True)
galaxy_test.dropna(how='any', inplace=True)

# Normalization of string values
galaxy_train['class'].replace(to_replace="GALAXY", value=1, inplace=True)
galaxy_train['class'].replace(to_replace="STAR", value=2, inplace=True)
galaxy_train['class'].replace(to_replace="QSO", value=3, inplace=True)

galaxy_test['class'].replace(to_replace="GALAXY", value=1, inplace=True)
galaxy_test['class'].replace(to_replace="STAR", value=2, inplace=True)
galaxy_test['class'].replace(to_replace="QSO", value=3, inplace=True)

features = ['u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field', 'plate', 'mjd', 'fiberid']
target = ['class']

train_X = galaxy_train[features]
train_y = galaxy_train[target]
test_X = galaxy_test[features]
test_y = galaxy_test[target]

train_y = np.ravel(train_y)


print("****AdaBoost****")

abc = AdaBoostClassifier(n_estimators=150)
abcmodel = abc.fit(train_X, train_y)
y_predABC = abcmodel.predict(test_X)

print("Accuracy:", accuracy_score(test_y, y_predABC))
print("Confusion matrix:\n", confusion_matrix(test_y, y_predABC))



print("\n****RANDOM FOREST CLASSIFIER****")

rfc = RandomForestClassifier(n_estimators=2048)
rfc.fit(train_X, train_y)
y_predRFC = rfc.predict(test_X)

print("Accuracy:", accuracy_score(test_y, y_predRFC))
print("Confusion matrix:\n", confusion_matrix(test_y, y_predRFC))



print("\n****RANDOM FOREST REGRESSOR****")

features_rfr = ['u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'class', 'field', 'plate', 'mjd', 'fiberid']

target_rfr = ['x_coord', 'y_coord', 'z_coord']

train_X_rfr = galaxy_train[features_rfr]
train_y_rfr = galaxy_train[target_rfr]
test_X_rfr = galaxy_test[features_rfr]
test_y_rfr = galaxy_test[target_rfr]

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(train_X_rfr, train_y_rfr)
y_predRFR = rfr.predict(test_X_rfr)

print("R2: ", r2_score(test_y_rfr, y_predRFR))
print("MSE: ", mean_squared_error(test_y_rfr, y_predRFR))



print("\n****MLP REGRESSOR****")

modelMLP = MLPRegressor(max_iter=350, batch_size=300)
modelMLP.fit(train_X_rfr, train_y_rfr)

y_predMLP = modelMLP.predict(test_X_rfr)


print("R2 = ", r2_score(test_y_rfr, y_predMLP))
print("MSE = ", mean_squared_error(test_y_rfr, y_predMLP))



