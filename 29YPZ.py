import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PowerTransformer,StandardScaler
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
def inverse_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)
df = pd.read_csv("Walmart.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y') 
df['day'] = df['Date'].dt.day
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df = df.drop("Date",axis=1)
X = df.drop('Weekly_Sales',axis=1)
y = df['Weekly_Sales']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)
scaled = StandardScaler()
X_train = scaled.fit_transform(X_train)
X_test = scaled.transform(X_test)
reg = LazyRegressor(verbose=0, 
                   ignore_warnings=True, 
                   custom_metric=None,
                   random_state=42)

# Tüm modelleri fit etme ve değerlendirme
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# Sonuçları görüntüleme
print("Tüm Modellerin Performans Karşılaştırması:")
print(models)