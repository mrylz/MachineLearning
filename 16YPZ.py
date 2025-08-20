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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("13-car_evaluation.csv")
df["vhigh"] = df["vhigh"].replace({
    "vhigh" : 0,
    "low" : 1,
    "med" : 2,
    "high" : 3
})
df["vhigh.1"] = df["vhigh.1"].replace({
    "vhigh" : 0,
    "low" : 1,
    "med" : 2,
    "high" : 3
})
df["2"] = df["2"].replace({
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5more" : 5
})
df["2.1"] = df["2.1"].replace({
    "2" : 2,
    "4" : 4,
    "more" : 5
})
df["small"] = df["small"].replace({
    "small" : 0,
    "med" : 1,
    "big" : 2
})
df["low"] = df["low"].replace({
    "low" : 0,
    "med" : 1,
    "high" : 2
})
df["unacc"] = df["unacc"].replace({
    "unacc" : 0,
    "acc" : 1,
    "vgood" : 2,
    "good" : 3
})
col_names = ["buying","maint","doors","persons","lug_boot","safety","class"]
df.columns = col_names
X = df.drop("class",axis=1)
y = df["class"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
model  = DecisionTreeClassifier()
param_grid = {
    "criterion": ["gini", "entropy", "log_loss"],   
    "max_depth": [None, 5, 10, 20, 30],             
    "min_samples_split": [2, 5, 10, 20],            
    "min_samples_leaf": [1, 2, 4, 6, 10],           
    "max_features": [None, "sqrt", "log2"],        
    "max_leaf_nodes": [None, 10, 20, 50, 100]       
}
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
    verbose=2
)
grid.fit(X_train_scaled,y_train)
y_pred = grid.best_estimator_.predict(X_test_scaled)
print("Hata Oranı : ",accuracy_score(y_test,y_pred))
print("en iyi parametreler : ",grid.best_params_)
print(classification_report(y_test,y_pred))