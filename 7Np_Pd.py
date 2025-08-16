import numpy as np
import pandas as pd
data_apply = pd.read_csv('8-apply_function_data.csv')
def Salary_category(salary):
    if salary < 50000:
        return "Low"
    elif  salary >= 50000 and salary < 80000:
        return "Medium"
    else:
        return "High"
data_apply["Salary_Category"] = data_apply["Salary"].apply(Salary_category)
print(data_apply)
