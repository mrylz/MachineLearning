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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier
from sklearn.impute import KNNImputer
df = pd.read_csv("16-diabetes.csv")
cols_with_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)
imputer = KNNImputer(n_neighbors=5) 
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
def drop_outliers_iqr_inplace(df, multiplier=1.5):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns   
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
    
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    return df
df = drop_outliers_iqr_inplace(df)
X = df.drop("Outcome",axis=1)
y = df["Outcome"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
reg = AdaBoostClassifier()
reg.fit(X_train_scaled,y_train)
y_pred = reg.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred)) 
