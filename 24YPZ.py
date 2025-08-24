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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor,GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
df = pd.read_csv("20-digitalskysurvey.csv")
df["class"] = df["class"].replace({
    "STAR" : 0,
    "GALAXY" : 1,
    "QSO" : 2
})
df = df.drop(["objid","specobjid","field","run","camcol","rerun"],axis=1)
#def flag_outliers_iqr(df, multiplier=3):
   # df = df.copy()
   # numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
   # for col in numeric_cols:
      #  Q1 = df[col].quantile(0.25)
     #  Q3 = df[col].quantile(0.75)
    #    IQR = Q3 - Q1
   #     lower = Q1 - multiplier * IQR
  #      upper = Q3 + multiplier * IQR
        
        # yeni kolon ekle: örn. "col_outlier"
 #       df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    
#    return df
#df = flag_outliers_iqr(df)
duplicates = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
X = df.drop("class",axis=1)
y = df["class"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = RobustScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
model = XGBClassifier()
model.fit(X_train_scaled,y_train)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test,y_pred))






