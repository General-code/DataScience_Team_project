from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("c:/work/application_record_minimum.csv")
x=df["FLAG_ABILITY"]
plt.title('0:Car X / Real Es. X \n0.5: One of them\n1: Both of them',fontsize=10)
plt.xlabel("ABILITY TO PAY BACK")
y=df["AMT_I0COME_TOTAL"]
plt.ylabel("ABILITY")
plt.plot(x,y,'o')
plt.show()