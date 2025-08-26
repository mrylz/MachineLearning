import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import boxcox
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.preprocessing import PowerTransformer,StandardScaler
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
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
# Power transformation
pt_x = PowerTransformer(method="yeo-johnson")
X_train = pt_x.fit_transform(X_train)
X_test = pt_x.transform(X_test)
y_train, lambda_y = boxcox(y_train)
xgb = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],  # 2 değer
    'max_depth': [3, 5],         # 2 değer  
    'learning_rate': [0.01, 0.1] # 2 değer
}
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=1,
    verbose=2,
    return_train_score=True
)
# Grid search'i çalıştırma
print("Grid Search başlatılıyor...")
grid_search.fit(X_train, y_train)

# Sonuçları gösterme
print("\nEn iyi parametreler:")
print(grid_search.best_params_)

print(f"\nEn iyi skor (R2): {grid_search.best_score_:.4f}")

# En iyi model ile tahmin
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred = inverse_boxcox(y_pred, lambda_y)
# Test sonuçları
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest Sonuçları:")
print("mse : ",mean_squared_error(y_test,y_pred))
print(f"R2 Score: {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f}")
# Basit bir kontrol
print("Gerçek değerler ve tahminler:")
comparison = pd.DataFrame({
    'Gerçek': y_test[:10],
    'Tahmin': y_pred[:10],
    'Fark': abs(y_test[:10] - y_pred[:10])
})
print(comparison)

print(f"\nTahmin edilen değerlerin range'i: {y_pred.min():.0f} - {y_pred.max():.0f}")
print(f"Gerçek değerlerin range'i: {y_test.min():.0f} - {y_test.max():.0f}")
# Ortalama göreceli hata
relative_errors = abs((y_test - y_pred) / y_test) * 100
mean_relative_error = relative_errors.mean()
print(f"Ortalama Göreceli Hata: {mean_relative_error:.2f}%")