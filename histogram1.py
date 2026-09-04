import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("Car_Data.csv")
plt.bar(data["Car_Data"],data["Price"],color=['red','green','blue','yellow','orange','pink']);
plt.title("Car Price")
plt.xlabel("Car")
plt.ylabel("Price")
plt.show() 

