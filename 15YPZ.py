import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv("12-health_risk_classification.csv")
mean = df["bmi_score"].mean()
std = df["bmi_score"].std()
z_scores = (df["bmi_score"] - mean) / std
outliers = df[np.abs(z_scores) > 3]
indis = outliers.index
df.drop(indis,inplace=True)
duplicates = df[df.duplicated(keep=False)]
#print("\nDuplicate satırlar (hepsi):")
#print(duplicates)
#sns.heatmap(df.corr(),annot=True)
#plt.show()
X = df.drop("high_risk_flag",axis=1)
y = df["high_risk_flag"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
model = KNeighborsClassifier() 
param_grid = {
    "n_neighbors" : [3,5,7,11],
    "algorithm" : ["ball tree","auto","kd_tree","brute"],
    "weights" : ["uniform","distance"]
}
reg = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs= 1,
    verbose=2
)
reg.fit(X_train_scaled,y_train)
y_pred = reg.best_estimator_.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(reg.best_params_)
