import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
import math

def inverse_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.exp(y)
    else:
        return np.power(y * lambda_ + 1, 1 / lambda_)

# Veri yükleme ve ön işleme
df = pd.read_csv("21-housing.csv")

# Kategorik değişkeni daha uygun şekilde kodlama (one-hot daha iyi olurdu)
df["ocean_proximity"] = df["ocean_proximity"].astype('category').cat.codes

# Aykırı değerleri temizleme ve kopyaları kaldırma
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

# Özellikler ve hedef
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train-test ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# Ölçeklendirme - DÜZELTİLDİ
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)  # Sadece transform!

# Power transformation
pt_x = PowerTransformer(method="yeo-johnson")
X_train_transformed = pt_x.fit_transform(X_train_scaled)
X_test_transformed = pt_x.transform(X_test_scaled)
# Hedef değişken dönüşümü
y_train_transformed, lambda_y = boxcox(y_train)

# Model eğitimi
reg = LGBMRegressor()
reg.fit(X_train_transformed, y_train_transformed)

# Tahmin ve değerlendirme
y_pred_tranformed = reg.predict(X_test_transformed)
y_pred_original = inverse_boxcox(y_pred_tranformed, lambda_y)

print("R² Score:", r2_score(y_test, y_pred_original))
print("MAE:", mean_absolute_error(y_test, y_pred_original))