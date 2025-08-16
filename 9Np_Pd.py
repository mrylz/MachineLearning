import numpy as np
import pandas as pd
apply_data = pd.read_csv('8-apply_function_data.csv')
def deneme(experience):
    if experience > 10:
        return 1
    else:
        return 0
apply_data["Adjusted"] = apply_data["Experience"].apply(deneme)    
apply_data["Performance_Experience"] = apply_data["Adjusted"] + apply_data["Performance_Score"]
print(apply_data)