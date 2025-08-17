import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
data = pd.read_csv('7-cyber_attack_data.csv')
X = data.drop('attack_type',axis=1)
y = data['attack_type']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
OvO = OneVsOneClassifier(LogisticRegression())
OvR = OneVsRestClassifier(LogisticRegression())
OvO.fit(X_train,y_train)
y_pred = OvO.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('Score : ' ,  score)
print(classification_report(y_pred,y_test))
OvR.fit(X_train,y_train)
y_pred = OvR.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('Score : ' ,  score)
print(classification_report(y_pred,y_test))