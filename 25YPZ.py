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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer,IterativeImputer
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
import lightgbm as lgb
df = sns.load_dataset("titanic")
df = df.drop(["deck","embark_town","alive"],axis=1)
df['age'] = df['age'].fillna(29.72)
df = df.dropna()
def label_encode_objects(df):
    df = df.copy()
    le = LabelEncoder()
    object_columns = df.select_dtypes(include=['object','category','bool']).columns  
    for col in object_columns:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
        except Exception as e:
            print(f"{col} sütununda hata: {e}")
    
    return df
df = label_encode_objects(df)
X = df.drop("survived",axis=1)
y = df["survived"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
reg = lgb.LGBMClassifier(verbosity=-1)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
y_pred_train = reg.predict(X_test)
print(classification_report(y_test,y_pred))
print(classification_report(y_test,y_pred_train))