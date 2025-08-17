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
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
reg = LogisticRegression()
penalty = ["l1","l2","elasticnet"]
c_values = [100,10,1,0.1,0.01]
solver = ["newton-cg","lbfgs","liblinear","sag","saga","newton-cholesky"]
params = dict(penalty=penalty,C = c_values,solver=solver)
cv = StratifiedKFold()
grid = GridSearchCV(estimator=reg,param_grid=params,cv=cv,scoring="accuracy",n_jobs=1)
grid.fit(X_resampled, y_resampled)
print(grid.best_params_)
y_pred = grid.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Score : ' ,  score)
print(classification_report(y_test,y_pred))
