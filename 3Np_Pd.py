import numpy as np
import pandas as pd
employee_csv = pd.read_csv("6-employee.csv")
#print(employee_csv)
#print(employee_csv.isna())
#print(employee_csv.describe())
#result = employee_csv.max()
#print(result)
datam = employee_csv.groupby("City")
print(datam["Salary"].mean())
print(datam["Experience"].mean())