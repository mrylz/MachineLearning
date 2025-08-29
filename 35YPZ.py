import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import outlier_utils as out
pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv("heart.csv")
df['RestingBP'] = df['RestingBP'].replace(0,130)
df['Cholesterol'] = df['Cholesterol'].replace(0,223)
df['Sex'] = df['Sex'].replace(
    {
        "M" : 1,
        "F": 0
    }
)
df['ChestPainType'] = df['ChestPainType'].replace(
    {
        "ASY" : 0,
        "NAP": 1,
        "ATA":2,
        "TA":3
    }
)
df['RestingECG'] = df['RestingECG'].replace(
    {
        "Normal" : 0,
        "LVH": 1,
        "ST":2
    }
)
df['ExerciseAngina'] = df['ExerciseAngina'].replace(
    {
        "N" : 0,
        "Y": 1
    }
)
df['ST_Slope'] = df['ST_Slope'].replace(
    {
        "Flat" : 2,
        "Up": 1,
        "Down":0
    }
)
print(out.detect_outliers(df, method='zscore', threshold=3))
df = out.remove_outliers(df)
print(df.info())