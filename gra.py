import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
df = pd.read_csv("12-house_energy_regression.csv")
mean = df["avg_indoor_temp_change"].mean()
std = df["avg_indoor_temp_change"].std()
z_scores = (df["avg_indoor_temp_change"] - mean) / std
outliers = df[np.abs(z_scores) > 3]
print(outliers)
duplicates = df[df.duplicated(keep=False)]
print(duplicates)
df = df.drop_duplicates()
