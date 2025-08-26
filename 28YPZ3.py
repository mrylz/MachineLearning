import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

# Veri yükleme ve ön işleme
df = pd.read_csv("21-housing.csv")

# Kategorik değişkeni daha uygun şekilde kodlama
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

# Sadece StandardScaler uygulama - Power Transformer YOK
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)  # Sadece transform!

# Model eğitimi - orijinal y değerleri üzerinde
reg = LGBMRegressor()
reg.fit(X_train_scaled, y_train)

# Tahmin ve değerlendirme
y_pred = reg.predict(X_test_scaled)

print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_absolute_error(y_test, y_pred)))