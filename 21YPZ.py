import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression,LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
df = pd.read_csv("17-cardekho.csv")
df = df.drop(["Unnamed: 0","car_name"],axis=1)
def remove_outliers_zscore(df, threshold):
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores > threshold).any(axis=1)
    df.drop(df[mask].index, inplace=True)
remove_outliers_zscore(df, threshold=3)   
duplicates = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
X = df.drop("selling_price",axis=1)
y = df["selling_price"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
encoder = TargetEncoder(smoothing=5.0)
X_train['brand'] = encoder.fit_transform(X_train['brand'], y_train)
X_test['brand'] = encoder.transform(X_test['brand'])
X_train['model'] = encoder.fit_transform(X_train['model'], y_train)
X_test['model'] = encoder.transform(X_test['model'])
X_train['seller_type'] = encoder.fit_transform(X_train['seller_type'], y_train)
X_test['seller_type'] = encoder.transform(X_test['seller_type'])
X_train['fuel_type'] = encoder.fit_transform(X_train['fuel_type'], y_train)
X_test['fuel_type'] = encoder.transform(X_test['fuel_type'])
X_train['transmission_type'] = encoder.fit_transform(X_train['transmission_type'], y_train)
X_test['transmission_type'] = encoder.transform(X_test['transmission_type'])
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
model = AdaBoostRegressor()
params = {
    "n_estimators" : [50, 80, 100, 120],
    "learning_rate" : [0.001, 0.01, 0.1, 1.0, 2.0],
    "loss" : ["linear", "square", "exponential"]
}
rcv = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='r2', cv=5,verbose=2)
rcv.fit(X_train_scaled, y_train)
y_pred = rcv.predict(X_test_scaled)
print("r2 score: ", r2_score(y_pred, y_test))
print("mean squared error: ", mean_squared_error(y_pred, y_test))
print("mean absolute error: ", mean_absolute_error(y_pred, y_test))



