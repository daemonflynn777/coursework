import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model

iris = sns.load_dataset("iris") # built-in dataset

dummies = pd.get_dummies(iris["species"], drop_first=False) # convert species to categorical values
iris = pd.concat([iris, dummies], axis=1) # concat this as a new row
iris.head()
