import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from imblearn.over_sampling import SMOTE
data = pd.read_csv('8-fraud_detection.csv')
X = data.drop('is_fraud',axis=1)
y = data['is_fraud']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
reg = LogisticRegression()
penalty = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.01]
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]
cls_weight = [{0 : w,1 : y} for w in[1,10,50,100] for y in [1,10,50,100]]
params = dict(penalty=penalty,C = c_values,solver=solver,class_weight = cls_weight)
cv = StratifiedKFold()
grid = GridSearchCV(estimator=reg,param_grid=params,cv=cv,scoring="accuracy",n_jobs=1)
grid.fit(X_train, y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Score : ' ,  score)
print(classification_report(y_test,y_pred))