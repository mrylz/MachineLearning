import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv("heart.csv")
df['RestingBP'] = df['RestingBP'].replace(0,130)
df['Cholesterol'] = df['Cholesterol'].replace(0,223)
df['Sex'] = df['Sex'].replace(
    {
        "M" : 1,
        "F": 0
    }
)
df['ChestPainType'] = df['ChestPainType'].replace(
    {
        "ASY" : 0,
        "NAP": 1,
        "ATA":2,
        "TA":3
    }
)
df['RestingECG'] = df['RestingECG'].replace(
    {
        "Normal" : 0,
        "LVH": 1,
        "ST":2
    }
)
df['ExerciseAngina'] = df['ExerciseAngina'].replace(
    {
        "N" : 0,
        "Y": 1
    }
)
df['ST_Slope'] = df['ST_Slope'].replace(
    {
        "Flat" : 2,
        "Up": 1,
        "Down":0
    }
)
df = df[df['Oldpeak'] >= 0]
X = df.drop("HeartDisease",axis=1)
y = df["HeartDisease"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
reg = ExtraTreesClassifier(n_estimators=100,max_depth=10,max_features='sqrt')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))