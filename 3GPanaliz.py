import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns
data = pd.read_csv("temizlenmis_veri.csv")
sns.barplot(x = "Installs" , y = "Category",data=data)
plt.show()