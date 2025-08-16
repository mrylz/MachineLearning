import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri yükleme
df = pd.read_csv("WineQT.csv")
plt.subplot(1,3,1)
sns.lineplot(x = 'quality' , y = 'alcohol' , data=df)
plt.xlabel("Quality")
plt.ylabel("Alcohol")
plt.title("Alcohol by Quality")
plt.subplot(1,3,2)
sns.lineplot(x = 'quality' , y = 'citric acid' , data=df)
plt.xlabel("QUALİTY")
plt.ylabel("citric acid")
plt.title("QUALİTY by Citric acid")
plt.subplot(1,3,3)
sns.lineplot(x = 'quality' , y = 'sulphates' , data=df)
plt.xlabel("Quality")
plt.ylabel("Sulphates")
plt.title("Quality by Sulphates")
plt.show()
