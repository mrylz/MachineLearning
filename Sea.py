import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("AKSEN Geçmiş Verileri.csv", decimal=",")
sns.heatmap(data.corr(numeric_only=True),annot = True)
plt.savefig("Aksen.png")
