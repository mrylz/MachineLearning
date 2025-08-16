import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sls
data = pd.read_csv("yeni_dosya.csv")
data['Rating'] = data['Rating'].fillna(data['Rating'].median())
data['Size'] = data['Size'].fillna(data['Size'].median())
data =  data.dropna()
print(data.isna().sum())
data.to_csv("temizlenmis_veri.csv", index=False)