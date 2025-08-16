import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv("2-multiplegradesdataset.csv")
X = data[["Study Hours","Sleep Hours","Attendance Rate","Social Media Hours"]]
y = data["Exam Score"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
regression = LinearRegression()
regression.fit(X_train, y_train)
#new_student = [
#    [0,5,40,2] 
#    ]
#result = scaler.transform(new_student)
#print(regression.predict(result))
y_pred = regression.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
score = r2_score(y_test,y_pred)
print("mse : " + str(mse) + "mae : " + str(mae) + "r2 : " + str(score))
