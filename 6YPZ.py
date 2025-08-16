import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.ensemble import ExtraTreesRegressor
data = pd.read_csv('Weather Station Locations.csv')
data.drop(['WBAN','NAME','STATE/COUNTRY ID','LAT','LON'],axis=1,inplace=True)
X = data[['Latitude','Longitude']]
y = data['ELEV']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#reg = LazyRegressor()
#models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#print(models)
model = ExtraTreesRegressor(
    n_estimators=200,      
    max_depth=None,        
    random_state=42,
    n_jobs=-1              
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)
student = [[39,8]]
result = scaler.transform(student)
print(model.predict(result))