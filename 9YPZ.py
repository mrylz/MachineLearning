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
reg.fit(X_resampled,y_resampled)
y_pred = reg.predict(X_test)
score = accuracy_score(y_pred,y_test)
print('Score : ' ,  score)
print(classification_report(y_pred,y_test))