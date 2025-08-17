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
import plotly.express as px 
data = pd.read_csv('9-seismic_activity_svm.csv')
print(data.columns)
data['under ^2'] = data['underground_wave_energy'] ** 2
data['vibra ^2'] = data['vibration_axis_variation'] ** 2
data['under * vibra'] = data['underground_wave_energy'] * data['vibration_axis_variation']
X = data.drop("seismic_event_detected",axis=1)
y = data["seismic_event_detected"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
fig = px.scatter_3d(data,x="under ^2",y="vibra ^2",z="under * vibra",color="seismic_event_detected")
fig.show()
reg = SVC(kernel="linear")
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print(classification_report(y_test,y_pred))