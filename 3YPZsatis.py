import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Veri yükleme
data = pd.read_csv('3-customersatisfaction.csv')
data.drop("Unnamed: 0", axis=1, inplace=True)

X = data[["Customer Satisfaction"]]
y = data["Incentive"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Polinom regresyon fonksiyonu
def poly_regression(degree):
    scaler = StandardScaler()
    poly_fea = PolynomialFeatures(degree=degree)
    lin_reg = LinearRegression()
    pipeline = Pipeline([
        ("Standard Scaler", scaler),
        ("poly_feat", poly_fea),
        ("lin_reg", lin_reg)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline.score(X_test, y_test)  # R² skorunu döndür

# Skorları tutacak liste
results = []

# 1’den 7’ye kadar polinom dereceleri için skorları bul
for degree in range(1,10):
    score = poly_regression(degree)
    results.append((degree, score))
    print(f"Degree={degree}, R² Score={score}")

# En iyi sonucu bul
best_degree, best_score = max(results, key=lambda x: x[1])
print(f"\nEn iyi sonuç: Degree={best_degree}, R²={best_score}")
