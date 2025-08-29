import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2"]
}

etc = ExtraTreesClassifier(random_state=42)
grid = GridSearchCV(etc, param_grid, cv=5, scoring="balanced_accuracy", n_jobs=1,verbose=2)
grid.fit(X_train, y_train)

print("En iyi parametreler:", grid.best_params_)
print("CV Balanced Accuracy:", grid.best_score_)
print("Test Skoru:", grid.score(X_test, y_test))
best_model = grid.best_estimator_

feature_names = df.drop("HeartDisease", axis=1).columns
importances = pd.Series(best_model.feature_importances_, index=feature_names)
importances.sort_values(ascending=True).plot(kind="barh", figsize=(8,6))
plt.title("Feature Importance - ExtraTreesClassifier")
plt.show()