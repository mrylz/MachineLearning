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
df = pd.read_csv("Iris.csv")
df.drop("Id",axis=1,inplace=True)
mean = df["PetalLengthCm"].mean()
std = df["PetalLengthCm"].std()
z_scores = (df["PetalWidthCm"] - mean) / std
outliers = df[np.abs(z_scores) > 3]
indis = outliers.index
df.drop(indis,inplace=True)
df["Species"] = df["Species"].replace({
    "Iris-setosa" : 1,
    "Iris-versicolor" : 2,
    "Iris-virginica" : 3
})
df.drop("PetalWidthCm",axis=1,inplace=True)
duplicates = df[df.duplicated(keep=False)]
df = df.drop_duplicates()
#print(df.info())
X = df.drop("Species",axis=1)
y = df["Species"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.transform(X_test)
model = LogisticRegression()
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],            # Düzenleme katsayısı
    "penalty": ["l1", "l2", "elasticnet", None],  # Ceza türü
    "solver": ["liblinear", "saga"],         # Çözücü
}
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=1,
    verbose=2
)
grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)
score = r2_score(y_test,y_pred)
print("hata : ",accuracy_score(y_test,y_pred))
print("En iyi parametreler : ",grid.best_params_)
print("r2 hata : ",score)