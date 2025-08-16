import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from lazypredict.Supervised import LazyClassifier,LazyRegressor

data = pd.read_csv('Summary of Weather.csv')
data.drop(["WindGustSpd","PoorWeather","DR","SPD","SND","FT","FB","FTI","ITH","PGT","TSHDSBRSGF","SD3","RHX","RHN","RVG","WTE"], axis=1, inplace=True)
data.dropna(inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data.drop('Date',axis=1,inplace=True)
data['Precip'] = data['Precip'].replace("T",np.nan)
data['Precip'] = data['Precip'].astype(float)
data['Precip'] = data["Precip"].fillna(data["Precip"].median())
data['Snowfall'] = data['Snowfall'].replace("#VALUE!",np.nan)
data['Snowfall'] = data['Snowfall'].astype(float)
data['Snowfall'] = data["Snowfall"].fillna(data["Snowfall"].median())
data['PRCP'] = data['PRCP'].replace("T",np.nan)
data['PRCP'] = data['PRCP'].astype(float)
data['PRCP'] = data["PRCP"].fillna(data["Precip"].median())
data['SNF'] = data['SNF'].replace("T",np.nan)
data['SNF'] = data['SNF'].astype(float)
data['SNF'] = data["SNF"].fillna(data["SNF"].median())
#sns.heatmap(data.corr(),annot=True)
#plt.show()
data = data.drop(['MAX','MIN','PRCP','SNF','YR','MO','DA','MEA'],axis=1)
#print(data.info())
#sns.heatmap(data.corr(),annot=True)
#plt.show()
X = data.drop('Precip',axis=1)
y = data["Precip"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#reg = LazyRegressor()
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#print(models)
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
print('Score : ', r2_score(y_test,y_pred))
print('mse : ' ,mean_squared_error(y_test,y_pred))
print('mae : ' ,mean_absolute_error(y_test,y_pred))
