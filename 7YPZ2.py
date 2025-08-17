import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
data = pd.read_csv('6-bank_customers.csv')
X = data.drop('subscribed',axis=1)
y = data['subscribed']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
log = LogisticRegression()
penalty = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.01]
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]
params = dict(penalty=penalty,C = c_values,solver=solver)
cv = StratifiedKFold()
grid = GridSearchCV(estimator=log,param_grid=params,cv=cv,scoring="accuracy",n_jobs=1)
grid.fit(X_train,y_train)
print(grid.best_params_)
y_pred = grid.predict(X_test)
score = accuracy_score(y_pred,y_test)
print(score)