import numpy as np
import pandas as pd
merge_data1 = pd.read_csv('7-merge_data1.csv')
merge_data2 = pd.read_csv('7-merge_data2.csv')
merge_data3 = pd.merge(merge_data1,merge_data2,on="Employee_ID",how='right')
print(merge_data2)