import numpy as np
import pandas as pd
apply_data = pd.read_csv('8-apply_function_data.csv')
apply_data["New_Name"] = apply_data["Name"].apply(lambda x : x.replace("_",""))
apply_data["Name"] = apply_data["New_Name"]
apply_data.drop("New_Name",axis=1,inplace=True)
print(apply_data)