import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
data = pd.read_csv('6-bank_customers.csv')
X = data.drop('subscribed',axis=1)
y = data['subscribed']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
log = LogisticRegression()
log.fit(X_train,y_train)
y_pred = log.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('Score : ' ,  score)
print(classification_report(y_pred,y_test))
#reg = LazyClassifier()
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#print(models)
