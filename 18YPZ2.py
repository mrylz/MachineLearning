import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("14-income_evaluation.csv")
col_names = ['age', 'workclass', 'finalweight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = col_names
df = df.dropna()
def remove_outliers_zscore(df, threshold):
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores > threshold).any(axis=1)
    df.drop(df[mask].index, inplace=True)
remove_outliers_zscore(df, threshold=5)   
duplicates = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
X = df.drop(['income'], axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
df.columns = col_names
# Örnek: tüm kategorik sütunları bul
cat_cols = X_train.select_dtypes(include=['object']).columns

# Her kategorik sütun için label encoding + boşluk temizleme
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    # boşluk silme ve NaN varsa doldurma
    X_train[col] = X_train[col].astype(str).str.strip().fillna("Unknown")
    X_train[col] = le.fit_transform(X_train[col])
    le_dict[col] = le   # ileride test setine de aynı encoder lazım
# Aynı işlemi test setine de uygula
for col in cat_cols:
    X_test[col] = X_test[col].astype(str).str.strip().fillna("Unknown")
    X_test[col] = le_dict[col].transform(X_test[col])
def corr_drop(X_test, threshold):
    columns_drop = set()
    corr = X_test.corr()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold:
                columns_drop.add(corr.columns[i])  
    df.drop(columns=columns_drop, axis=1, inplace=True)
corr_drop(X_test, 0.90)
scaled = RobustScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
reg = RandomForestClassifier()
rf_params = {"max_depth": [None, 10, 20, 30, 50],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "min_samples_leaf" : [1, 2,3,4], 
             "n_estimators": [100, 200, 500, 1000]}
rfc = RandomForestClassifier()
rscv = RandomizedSearchCV(estimator=rfc, param_distributions=rf_params,n_iter=10,cv=3,verbose=2,n_jobs=1)
rscv.fit(X_train, y_train)
y_pred = rscv.predict(X_test) 
print(f'Model accuracy score with default decision-trees : {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test,y_pred))