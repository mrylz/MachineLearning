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
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
df = pd.read_csv("15-gym_crowdedness.csv")
df =  df.drop(["date","timestamp"],axis=1)
def remove_outliers_zscore(df, threshold):
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask = (z_scores > threshold).any(axis=1)
    df.drop(df[mask].index, inplace=True)
remove_outliers_zscore(df, threshold=5) 
duplicates = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
print(df.info())
X = df.drop("number_people",axis=1)
y = df["number_people"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 15)
scaled = RobustScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
model = RandomForestRegressor() 
param_dist = {
    'n_estimators': [100, 200, 300, 500, 800, 1000],     # Ağaç sayısı
    'max_depth': [None, 10, 20, 30, 50],                 # Maksimum derinlik
    'min_samples_split': [2, 5, 10],                     # Bölünme kriteri
    'min_samples_leaf': [1, 2, 4],                       # Yapraktaki min örnek
    'max_features': ['auto', 'sqrt', 'log2'],            # Özellik seçimi
    'bootstrap': [True, False]                           # Bootstrap kullanımı
}
rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,                   # Denenecek rastgele kombinasyon sayısı
    scoring='r2',                # Alternatif: 'neg_mean_squared_error'
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=1
)
random_search.fit(X_train_scaled, y_train)
y_pred = random_search.predict(X_test)
print(r2_score(y_test,y_pred))