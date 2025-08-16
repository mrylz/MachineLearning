import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
data = pd.read_csv('2-temizlenmis_forest.csv')
X = [["temperature","rh","rain","ffmc","dmc","dc","isi","bui"]]
y = ["fwi"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regression = LinearRegression()
regression.fit(X_train,y_train)
print("Coef : ",regression.coef_)
print("intercept : ",regression.intercept_)
