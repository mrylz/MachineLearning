import numpy as np
import pandas as pd
concat_data1 = pd.read_csv('7-concat_data1.csv')
concat_data2 = pd.read_csv('7-concat_data2.csv')
concat_data3 = pd.concat([concat_data1,concat_data2],ignore_index=True)
print(concat_data3.head())

