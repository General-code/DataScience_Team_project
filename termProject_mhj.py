import numpy as np
import pandas as pd

credit = pd.read_csv(r"C:\Users\82103\github\DS\credit_record.csv")
application = pd.read_csv(r"C:\Users\82103\github\DS\application_record.csv")

#combine application_record.csv & credit_record.csv() 
df = application.merge(credit,how='left',on='ID')
print("\n============combien two csv file===============")
print(df)#it must have na becuase application_record doesn't have a lot of ids in the credit_record.csv
df = df[df['STATUS'].notna()]

print("\n============STATUS null column remove(ID that doesn't exist in application=======")
print(df)#ID overlapping between credit and application

#Remove duplicate IDs except the first one
df = df.drop_duplicates(['ID'],keep="first",ignore_index=True)

df2 = df[df['STATUS']=='C']#paid off that month -> 17950개
df3 = df[df['STATUS']=='X']#No loan for the month -> can except (9669개)
df4 = df[df['STATUS']=='0']#1-29 days past due -> 8456개
df5 = df[df['STATUS']=='1']#30-59 days past due -> 276개
df6 = df[df['STATUS']=='2']#60-89 days overdue -> 19개
df7 = df[df['STATUS']=='3']#90-119 days overdue -> 6개
df8 = df[df['STATUS']=='4']#120-149days overdue -> 5개
df9 = df[df['STATUS']=='5']#Overdue or bad debts, write-offs for more than 150 days ->76개
#print(len(df2),"_",len(df3),"_",len(df4),"_",len(df5),"_",len(df6),"_",len(df7),"_", len(df8) ,"_", len(df9))
print("\n===============remove ID duplicate================")
print(df)

#Remove STATUS=='X' (it does't need to analysis)
idx = df[df['STATUS']=='X'].index
df.drop(idx,inplace=True)
print("\n========remvoe 'STAUS=X'=============")
print(df)

#Remove two columns -> FLAG_MOBIL/OCCUPATION_TYPE
df.drop(['FLAG_MOBIL' , 'OCCUPATION_TYPE'], axis=1, inplace=True)
print("\n========column list=============")
print(df.columns.tolist())


#연체 없는 사람과 있는 사람으로 분류하여 라벨링

#df.loc[df['STATUS']=='C']=1
df.at[df['STATUS']=='C', 'STATUS']=0
df.at[df['STATUS']=='0', 'STATUS']=1
df.at[df['STATUS']=='1', 'STATUS']=1
df.at[df['STATUS']=='2', 'STATUS']=1
df.at[df['STATUS']=='3', 'STATUS']=1
df.at[df['STATUS']=='4', 'STATUS']=1
df.at[df['STATUS']=='5', 'STATUS']=1
print("\n========결측값 유무로 STATUS 재분류=============")
print(df)

#fill na -> 이미 null값은 NaN으로 되어 있긴함
df.fillna('NaN',inplace=True)

#categorical feature
#NAME_HOUSING_TYPE/ CODE_GENDER/ FLAG_OWN_CAR
#/NAME_INCOME_TYPE/NAME_FAMILY_STATUS/NAME_EDUCATION_TYPE
#STATUS <- 이미 라벨링

#label encoding
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()

df['NAME_HOUSING_TYPE']=le.fit_transform(df['NAME_HOUSING_TYPE'])
df['NAME_INCOME_TYPE']=le.fit_transform(df['NAME_INCOME_TYPE'])
df['NAME_FAMILY_STATUS']=le.fit_transform(df['NAME_FAMILY_STATUS'])
df['NAME_EDUCATION_TYPE']=le.fit_transform(df['NAME_EDUCATION_TYPE'])
df['CODE_GENDER']=le.fit_transform(df['CODE_GENDER'])
df['FLAG_OWN_CAR']=le.fit_transform(df['FLAG_OWN_CAR'])

#outlier

#numericla features(those have outliers)
def outliers_iqr(df,name):
    q1 = df[name].quantile(0.25)
    q3 = df[name].quantile(0.75)
    iqr = q3 - q1
    search = df[(df[name]<(q1 - 1.5 * iqr))|(df[name]>(q3 + 1.5 * iqr))]
    df = df.drop(search.index,axis=0)
    return df

df = outliers_iqr(df,'AMT_INCOME_TOTAL')
df = outliers_iqr(df,'DAYS_BIRTH')
df = outliers_iqr(df,'DAYS_EMPLOYED')

#feature scaling (categorical은 labeling 돼서 필요X)
#'AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED' 세가지만

cols = {'AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED'}

scaler = preprocessing.StandardScaler()
scaled_Sd = scaler.fit_transform(df.loc[:,cols])
df.loc[:,cols] = scaled_Sd

print("\n========Standard Scaler for some numerical features=============")
print(df[['AMT_INCOME_TOTAL', 'DAYS_BIRTH','DAYS_EMPLOYED']])









