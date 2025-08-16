import numpy as np
import pandas as pd
weather_df = pd.read_excel('6-weather.xlsx')
#print(weather_df.head())
#print(weather_df.tail())
#print(weather_df.info())
#print(weather_df.describe())
#print(weather_df.count())
print(weather_df.isna())