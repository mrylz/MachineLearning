import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('temiz_bmw.csv')
sns.lineplot(x= 'Fuel_Type' , y= 'Sales_Classification' ,data=data)
plt.xlabel('Sales')
plt.ylabel('Price')
plt.title('Sales by Price')
plt.show()
