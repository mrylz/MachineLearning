import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sls
data = pd.read_csv("17-googleplaystore.csv")
data = data.drop(index=10472)
data['Reviews'] = data['Reviews'].astype(int)
data['Reviews'] = (data['Reviews'] / 100000).astype(int)
data['Size'] = data['Size'].str.replace("M","000")
data['Size'] = data['Size'].str.replace("k","")
data['Size'] = data['Size'].replace("Varies with device",np.nan)
data['Size'] = data['Size'].astype(float)
data['Installs'] = data['Installs'].str.replace("+","")
data['Installs'] = data['Installs'].str.replace(",","")
data['Installs'] = data['Installs'].str.replace("$","")
data['Installs'] = data['Installs'].astype(float)
data['Installs'] = (data['Installs'] / 100000).astype(float)
data['Price'] = data['Price'].str.replace("$","")
data['Price'] = data['Price'].astype(float)
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
data['Day'] = data['Last Updated'].dt.day
data['Month'] = data['Last Updated'].dt.month
data['Year'] = data['Last Updated'].dt.year
data.drop_duplicates(subset=['App'],keep='first',inplace=True)
data.to_csv("yeni_dosya.csv", index=False)

