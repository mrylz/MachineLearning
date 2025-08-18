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
#mean = df["PetalLengthCm"].mean()
#std = df["PetalLengthCm"].std()
#z_scores = (df["PetalWidthCm"] - mean) / std
#outliers = df[np.abs(z_scores) > 3]
#print(outliers)
#indis = outliers.index
#df.drop(indis,inplace=True)
#print(df["Species"].unique())
#print(df["Species"].value_counts())
df["Species"] = df["Species"].replace({
    "Iris-setosa" : 1,
    "Iris-versicolor" : 2,
    "Iris-virginica" : 3
})
df.drop("PetalWidthCm",axis=1,inplace=True)
#print(df.info())
#duplicates = df[df.duplicated(keep=False)]
#print("\nDuplicate satırlar (hepsi):")
#print(duplicates)
# Duplicate satırlardan sadece birini bırakıp diğerlerini silme
#df_cleaned = df.drop_duplicates()
#print("\nDuplicate'lardan temizlenmiş DataFrame:")
#print(df_cleaned.info())
X = df.drop("Species",axis=1)
y = df["Species"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)
scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.transform(X_test)
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
