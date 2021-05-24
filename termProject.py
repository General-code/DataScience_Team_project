import numpy as np
import pandas as pd

credit = pd.read_csv("D:/credit_record.csv")
application = pd.read_csv("D:/application_record.csv")

df = application.merge(credit,how='left',on='ID')
df = df[df['STATUS'].notna()]
df2 = df[df['STATUS']=='C']
df3 = df[df['STATUS']=='X']
df4 = df[df['STATUS']=='0']
df5 = df[df['STATUS']=='1']
df6 = df[df['STATUS']=='2']
df7 = df[df['STATUS']=='3']
print(len(df2),"_",len(df3),"_",len(df4),"_",len(df5),"_",len(df6),"_",len(df7))