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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
df = pd.read_csv("18-concrete_data.csv")
def remove_outliers_zscore(df, threshold):
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores > threshold).any(axis=1)
    df.drop(df[mask].index, inplace=True)
remove_outliers_zscore(df, threshold=5)   
duplicates = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
X = df.drop("Strength",axis=1)
y = df["Strength"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train_scaled =  scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
params = {
        "n_estimators" : [100, 150, 200],
        "max_depth" : [3,4,5],
        "loss" : ["squared_error", "absolute_error", "huber", "quantile"],
        "learning_rate" : [0.01, 0.1, 0.5]
}
model = GradientBoostingRegressor()
reg = RandomizedSearchCV(estimator=GradientBoostingRegressor(),scoring="r2",param_distributions=params, cv=5,verbose=2)
reg.fit(X_train_scaled,y_train)
y_pred = reg.predict(X_test_scaled)
print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

