import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import r2_score
data = pd.read_csv('4-Algerian_forest_fires_dataset.csv')
data =data.dropna().reset_index()
data.drop(data[~data['day'].str.isnumeric()].index,inplace=True)
data.drop("day", axis=1, inplace=True)
data.drop("month", axis=1, inplace=True)
data.drop("year", axis=1, inplace=True)
data['Temperature'] = data['Temperature'].astype(float)
data[' RH'] = data[' RH'].astype(float)
data[' Ws'] = data[' Ws'].astype(float)
data['Rain '] = data['Rain '].astype(float)
data['FFMC'] = data['FFMC'].astype(float)
data['DMC'] = data['DMC'].astype(float)
data['DC'] = data['DC'].astype(float)
data['ISI'] = data['ISI'].astype(float)
data['BUI'] = data['FWI'].astype(float)
data['FWI'] = data['FWI'].astype(float)
data['Classes  '] = data['Classes  '].str.replace(" ","")
data['Classes  '] = data['Classes  '].str.replace("notfire","0")
data['Classes  '] = data['Classes  '].str.replace("fire","1")
data['Classes  '] = data['Classes  '].astype(int)
data.columns = data.columns.str.strip().str.lower()
X = data.drop("fwi",axis=1)
y = data["fwi"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
def corr_drop(data,threshold):
    columns_drop = set()
    corr = data.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j]) > threshold:
                columns_drop.add(corr.columns[i])
    return columns_drop      
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linear = Lasso()
linear.fit(X_train_scaled,y_train)
y_pred = linear.predict(X_test_scaled)
print(r2_score(y_test,y_pred))
