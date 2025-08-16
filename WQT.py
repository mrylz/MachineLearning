import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('WineQT.csv')
plt.subplot(1,2,1)
sns.lineplot(x = "alcohol", y = "quality" , data=data)
plt.title("Quality by Alcohol")
plt.xlabel("ALCOHOL")
plt.ylabel("QUALİTY")
plt.subplot(1,2,2)
sns.lineplot(x = "volatile acidity", y = "quality" , data=data)
plt.title("Quality by volatile acidity")
plt.xlabel("SULPHATES")
plt.ylabel("QUALİTY")
plt.show()
#print(data.columns)