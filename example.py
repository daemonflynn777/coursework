import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def rmse(y_pred, y_true) :
    return np.sqrt(mean_squared_error(y_pred, y_true))


iris = sns.load_dataset("iris") # built-in dataset

dummies = pd.get_dummies(iris["species"], drop_first=True) # convert species to categorical values
iris = pd.concat([iris, dummies], axis=1) # concat this as new columns
print(iris.head())
#print(dummies)

X = iris[iris.columns.difference(["petal_width", "species"])]
X = sm.add_constant(X)
y = iris["petal_width"]
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

ypred = results.predict(X)
print('\nRmse is %.2f' %rmse(ypred, y))
print('Mae is %.2f' %mean_absolute_error(ypred, y))
print('R2 is %.2f' %r2_score(ypred, y))
