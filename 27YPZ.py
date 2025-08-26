import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import math
from lightgbm import LGBMRegressor
from scipy.stats import boxcox
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
df = pd.read_csv("24-medical_cost.csv")
df.drop("Id", inplace=True, axis=1)
df["sex"] = df["sex"].map({"male" : 0, "female": 1})
df["smoker"] = df["smoker"].map({"no" : 0, "yes": 1})
X = df.drop("charges", axis=1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)
def inverse_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)
categorical_cols = ["region"]
preprocessor = ColumnTransformer(transformers=
    [
       ('cat', OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ], remainder= "passthrough"
     )
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)    
y_train_transformed, lambda_y = boxcox(y_train)
model = LGBMRegressor()
model.fit(X_train, y_train_transformed)
y_pred_transformed = model.predict(X_test)
y_pred_original = inverse_boxcox(y_pred_transformed, lambda_y)
print(r2_score(y_pred_original, y_test))
print(mean_squared_error(y_pred_original, y_test))