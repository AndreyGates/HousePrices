from asyncio.windows_events import NULL
from cmath import nan
from numpy import NaN
import pandas as pd
import numpy as np

df = pd.read_csv('../HousePrices/src/train.csv')

df = df.dropna(axis=1, thresh=len(df.values)/1.5) # dropping columns where 1/3 is Nan or more)
df = df.fillna(df.mean()) # imputing missing numerical values with feature means

#df = df.drop(1379) # df[1379]['Electrical'] is the only empty string in the column

''''''
null_columns=df.columns[df.isnull().any()]
print(df.isnull().sum()[null_columns]) # only categorical nulls
''''''

