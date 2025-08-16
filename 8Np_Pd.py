import numpy as np
import pandas as pd

apply_data = pd.read_csv('8-apply_function_data.csv')

# Fonksiyon: sadece score değerini alıp artırmalı
def performance_points(row):
    if row["Experience"] > 10:
        return row["Performance_Score"] + 1
    else:
        return row["Performance_Score"]

# apply ile satır bazında (axis=1) işle
apply_data["Performance_experience"] = apply_data.apply(performance_points, axis=1)

# Sonuçları yazdır
print(apply_data)
