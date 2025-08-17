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
from sklearn.svm import SVC
data = pd.read_csv('9-loan_risk_svm.csv')
print(data.columns)
X = data.drop('loan_risk',axis=1)
y = data['loan_risk']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
reg = SVC(kernel="rbf")
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(classification_report(y_test,y_pred))
