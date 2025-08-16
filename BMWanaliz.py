import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('BMW_Car_Sales_Classification.csv')
data['Transmission'] = data['Transmission'].map({'Automatic': 1, 'Manual': 0})
data['Sales_Classification'] = data['Sales_Classification'].map({'High': 1, 'Low': 0})
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'Hybrid': 2, 'Electric': 3})
data['Region'] = data['Region'].map({'South America': 0, 'Africa': 1, 'Europe': 2, 'North America': 3,'Middle East' : 4,'Asia' : 5})
#freq_encoding = data['Color'].value_counts()
data.to_csv("temiz_bmw.csv", index=False)


