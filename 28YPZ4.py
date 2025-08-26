import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from scipy.stats import boxcox

def inverse_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)

# Veri yükleme ve ön işleme (ortak kısım)
df = pd.read_csv("21-housing.csv")
df["ocean_proximity"] = df["ocean_proximity"].astype('category').cat.codes

def drop_outliers_iqr_inplace(df, multiplier=3):
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
df = df.drop_duplicates()

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# 1. Hiç dönüşüm yok
print("=== HİÇ DÖNÜŞÜM YOK ===")
reg1 = LGBMRegressor()
reg1.fit(X_train, y_train)
y_pred1 = reg1.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred1))
print("MAE:", mean_absolute_error(y_test, y_pred1))
print()

# 2. Sadece StandardScaler
print("=== SADECE STANDARDSCALER ===")
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

reg2 = LGBMRegressor()
reg2.fit(X_train_scaled, y_train)
y_pred2 = reg2.predict(X_test_scaled)
print("R² Score:", r2_score(y_test, y_pred2))
print("MAE:", mean_absolute_error(y_test, y_pred2))
print()

# 3. Power Transformer ile (orijinal kodunuz - düzeltilmiş)
print("=== POWER TRANSFORMER İLE ===")
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)

pt_x = PowerTransformer(method="yeo-johnson")
X_train_transformed = pt_x.fit_transform(X_train_scaled)
X_test_transformed = pt_x.transform(X_test_scaled)

y_train_transformed, lambda_y = boxcox(y_train)

reg3 = LGBMRegressor()
reg3.fit(X_train_transformed, y_train_transformed)
y_pred_tranformed = reg3.predict(X_test_transformed)
y_pred3 = inverse_boxcox(y_pred_tranformed, lambda_y)

print("R² Score:", r2_score(y_test, y_pred3))
print("MAE:", mean_absolute_error(y_test, y_pred3))