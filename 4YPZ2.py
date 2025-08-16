import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv('temizlenmis_forest.csv')
data.drop("day", axis=1, inplace=True)
data.drop("month", axis=1, inplace=True)
data.drop("year", axis=1, inplace=True)
data.drop(" Ws", axis=1, inplace=True)
data.drop("Classes  ", axis=1, inplace=True)
data.columns = data.columns.str.strip().str.lower()
print(data.info())
data.to_csv("2-temizlenmis_forest.csv", index=False)
