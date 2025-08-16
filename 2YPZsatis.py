import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Veri yükleme
data = pd.read_csv('3-customersatisfaction.csv')
data.drop("Unnamed: 0", axis=1, inplace=True)

X = data[["Customer Satisfaction"]]
y = data["Incentive"]

# Veri setini böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Polinomial özellikler
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Modeli eğit
regression = LinearRegression()
regression.fit(X_train_poly, y_train)

# Tahmin
y_pred = regression.predict(X_test_poly)

# Performans
score = r2_score(y_test, y_pred)
print("R2 Skoru:", score)

# Görselleştirme
plt.scatter(X_test, y_test, color="red", label="Gerçek Değerler")
plt.scatter(X_test, y_pred, color="blue", label="Tahminler")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Incentive")
plt.legend()
plt.show()
