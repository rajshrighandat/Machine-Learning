#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

#from sklearn.preprocessing import StandardScaler,StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
df.info()
df.head() 

#print the duplicate rows

df.duplicated().sum()

#print last 5
print(df.tail())

#print first 5
print(df.head())

#print rows by column
print(df.shape)

#print the no. of column  and rows,dta type ,missing value
print(df.info())

#to check missing value
print(df.isnull().sum())

#to check the data type
print(df.dtypes)

# Replace all NaN values with 0
df.fillna(value=0, inplace=True)

# Drops all rows that contain at least one missing value (NaN)
df_clean = df.dropna()

df_clean = df.drop(columns='sepal_width')

# Remove the row at index position 0
df_clean = df.drop(0)

# Remove multiple specific rows
df_clean = df.drop([1, 2])

