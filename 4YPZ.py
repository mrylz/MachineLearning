import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('4-Algerian_forest_fires_dataset.csv')
data =data.dropna()
data.drop(data[~data['day'].str.isnumeric()].index,inplace=True)
data['day'] = data['day'].astype(int)
data['month'] = data['month'].astype(int)
data['year'] = data['year'].astype(int)
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
data.to_csv("temizlenmis_forest.csv", index=False)
