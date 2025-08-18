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
from sklearn.svm import SVC,SVR
import plotly.express as px 
data = pd.read_csv('10-diamonds.csv')
data.drop("Unnamed: 0",axis=1,inplace=True)
data["cut"] = data["cut"].replace({
    "Ideal": 1,
    "Premium": 2,
    "Very Good": 3,
    "Good": 4,
    "Fair": 5
})
data["color"] = data["color"].replace({
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "H": 5,
    "I": 6,
    "J": 7
})
data["clarity"] = data["clarity"].replace({
    "IF": 1,
    "VVS1": 2,
    "VVS2": 3,
    "VS1": 4,
    "VS2": 5,
    "SI1": 6,
    "SI2": 7,
    "I1": 8,
    "I2": 9,
    "I3": 10
})
data = data.drop(data[data["x"] == 0].index)
data = data.drop(data[data["y"] == 0].index)
data = data.drop(data[data["z"] == 0].index)
#print(data.describe())
#print(data.info())
#print(data["clarity"].unique())
data.drop(["x","y","z"],axis=1,inplace=True)
#print(data.columns)
#sns.heatmap(data.corr(),annot=True)
#plt.show()
X = data.drop("price",axis=1)
y = data["price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.transform(X_test)
reg = SVR(kernel="rbf",C=1000,gamma="auto")
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_test,y_pred)
print(score)
print("mae",mean_absolute_error(y_test,y_pred))