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
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
df = pd.read_csv("financial_regression.csv")
df.drop(["date","us_rates_%"],axis=1,inplace=True)

X = df.drop("gold open",axis=1)
y = df["gold open"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
model  = DecisionTreeRegressor()
#param_grid = {
    #"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],   
    #"max_depth": [None, 5, 10, 20, 30],
    #"splitter" : ["best", "random"],             
    #"min_samples_split": [2, 5, 10, 20],            
   # "min_samples_leaf": [1, 2, 4, 6, 10],           
  #  "max_features": [None, "sqrt", "log2"],        
 #   "max_leaf_nodes": [None, 10, 20, 50, 100]       
#}
#grid = GridSearchCV(
   # estimator=model,
   # param_grid=param_grid,
   # cv=5,
   # scoring="r2",
   # n_jobs=1,
   # verbose=2
#)
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
print("Hata Oranı : ",r2_score(y_test,y_pred))
#print("en iyi parametreler : ",grid.best_params_)
print(mean_absolute_error(y_test,y_pred))
