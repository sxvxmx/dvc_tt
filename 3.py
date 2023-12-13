import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn
import sklearn
import yaml
import math
import seaborn as sns
import pickle as pkl

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

data_test = pd.read_csv("data/clean/data_test_prepared.csv")
data_train = pd.read_csv("data/clean/data_train_prepared.csv")
target = pd.read_csv("data/clean/target.csv")

data_train = data_train.drop("Unnamed: 0",axis=1)
data_test = data_test.drop("Unnamed: 0",axis=1)
target = target.drop("Unnamed: 0",axis=1)

gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(data_train, target)

out = pd.DataFrame({"pred":gbc_clf.predict(data_test)})
pkl.dump(gbc_clf, open("models/model.pkl", 'wb'))
out.to_csv("data/out/out.csv")