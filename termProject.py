import time

import sklearn
from sklearn import preprocessing, __all__, linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns',None)

credit = pd.read_csv(r"credit_record.csv")
application = pd.read_csv(r"application_record.csv")
#combine application_record.csv & credit_record.csv()
df = application.merge(credit,how='left',on='ID')
print("\n============combien two csv file===============")
print(df)#it must have na becuase application_record doesn't have a lot of ids in the credit_record.csv
df = df[df['STATUS'].notna()]

# make DAYS_BIRTH and DAYS_EMPLOYED be Non-negative value
df["DAYS_BIRTH"] = -df["DAYS_BIRTH"]
df["DAYS_EMPLOYED"] = -df["DAYS_EMPLOYED"]

print("\n============STATUS null column remove(ID that doesn't exist in application=======")
print(df)#ID overlapping between credit and application

#Remove duplicate IDs except the first one
df = df.drop_duplicates(['ID'],keep="first",ignore_index=True)
print("\n===============remove ID duplicate================")
print(df)

#FLAG_MOBIL romove reason
df_mobile = df[(df['FLAG_MOBIL']==1)]
mobile_ratio = len(df_mobile)/len(df)
print("\n==========Ratio of mobile phone holers===========")
print("Ratio of mobile phone holers : " , mobile_ratio,"\n")

#OCCUPATION remove reason
ocp_null = df['OCCUPATION_TYPE'].isnull().sum() #null값의 개수를 구하기
print("ocp_null : " , ocp_null)
ocp_len = len(df)
print("occupation_type percentage : ", ocp_null/ocp_len,"\n")

#Remove two columns -> FLAG_MOBIL/OCCUPATION_TYPE(모두 동일한 값이라서, 0.3%의 null값이 있어)
df.drop(['FLAG_MOBIL' , 'OCCUPATION_TYPE'], axis=1, inplace=True)
print("\n========column list(after remove two columns=============")
print(df.columns.tolist())

#create Random null values
import random
for i in range(1,1000):
    random_row = random.randrange(0,36456)
    random_col = 4
    df.iloc[random_row,random_col] = np.nan
print("The number of null value in CNT_CHILDREN(After) ", df['CNT_CHILDREN'].isnull().sum())

#show boxplot to find if there are outliers
plt.figure(figsize=(12,8))
sns.boxplot(data=df['DAYS_BIRTH'],color='red')
plt.show()
sns.boxplot(data=df['DAYS_EMPLOYED'],color='red')
plt.show()

# minus number in DAYS_EMPLOYED means the person is unemployed
df.loc[df['DAYS_EMPLOYED']<=0,'DAYS_EMPLOYED'] = 0
sns.boxplot(data=df['DAYS_EMPLOYED'],color='red')
plt.show()


#categorical feature
#NAME_HOUSING_TYPE/ CODE_GENDER/ FLAG_OWN_CAR
#/NAME_INCOME_TYPE/NAME_FAMILY_STATUS/NAME_EDUCATION_TYPE
#STATUS <- 이미 라벨링

#label encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels

labelEncode(df,'CODE_GENDER')
labelEncode(df,'FLAG_OWN_CAR')
labelEncode(df,'FLAG_OWN_REALTY')
labelEncode(df,'STATUS')


#onehot encoding
from sklearn import preprocessing
le = preprocessing.OneHotEncoder()
def onehotEncode(df, name):
   le = preprocessing.OneHotEncoder(handle_unknown='ignore')
   enc = df[[name]]
   enc = le.fit_transform(enc).toarray()
   enc_df = pd.DataFrame(enc, columns=le.categories_[0])
   df.loc[:, le.categories_[0]] = enc_df

#feature scaling (categorical은 labeling 돼서 필요X)
#'AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED' 세가지만

cols = {'DAYS_BIRTH','DAYS_EMPLOYED'}

scaler = preprocessing.StandardScaler()
scaled_Sd = scaler.fit_transform(df.loc[:,cols])
df_Sd = df
df_Sd.loc[:,cols] = scaled_Sd

print("\n========Standard Scaler for some numerical features=============")
print(df[['DAYS_BIRTH','DAYS_EMPLOYED']])



#outlier
#numericla features(those have outliers)
def outliers_iqr(df,name):
    q1 = df[name].quantile(0.25)
    q3 = df[name].quantile(0.75)
    iqr = q3 - q1
    search = df[(df[name]<(q1 - 1.5 * iqr))|(df[name]>(q3 + 1.5 * iqr))]
    df = df.drop(search.index,axis=0)
    return df

df = outliers_iqr(df,'DAYS_BIRTH')
df = outliers_iqr(df,'DAYS_EMPLOYED')




# Correlation HeatMap
# Use only continuous value
# exclude non-necessary features
dfc = df.drop(columns=["FLAG_PHONE", "STATUS", "ID"])
plt.figure(figsize=(14, 14))
sns.heatmap(data=dfc.corr(), annot=True,
fmt = '.2f', linewidths=.5, cmap='Blues')
plt.show()

# plot linear regression between AMT_INCOME and the other related features
# Use subplots with thw rows and four columns. axs has 4x2 ax.
sampled_df = df.sample(n=150, random_state=1)
fig, axs = plt.subplots(figsize=(16, 8), ncols=4, nrows=2)
lm_features = ["CNT_FAM_MEMBERS", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_WORK_PHONE",
               "MONTHS_BALANCE", "CNT_CHILDREN", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
for i, feature in enumerate(lm_features):
    row = int(i / 4)
    col = i % 4
    sns.regplot(x=feature, y='AMT_INCOME_TOTAL', data=sampled_df, ax=axs[row][col])
plt.show()

#fill 'CNT_CHILDREN' with predicted value with 'CNT_FAM_MEMBERS'
#위에 나온 correlation에서 CNT_FAM_MEMBERS가 CNT_CHILDREN과 0.9로 아주 높은 상관관계를 가졌기 때문

df_dropNa = df.dropna()
x_gr = df_dropNa[['CNT_FAM_MEMBERS']]
y_gr = df_dropNa['CNT_CHILDREN']
reg = LinearRegression().fit(x_gr,y_gr)
child_pred = reg.predict(df[['CNT_FAM_MEMBERS']])
print("The number of null value in CNT_CHILDREN(Before) ",df['CNT_CHILDREN'].isnull().sum())
df['CNT_CHILDREN'] = np.where(df['CNT_CHILDREN'].isnull(),pd.Series(child_pred.flatten()),df['CNT_CHILDREN'])
print("The number of null value in CNT_CHILDREN(After) ",df['CNT_CHILDREN'].isnull().sum())


#encode with onehotEncode
df = df.reset_index(drop=True)
onehotEncode(df,'NAME_HOUSING_TYPE')
onehotEncode(df,'NAME_INCOME_TYPE')
onehotEncode(df,'NAME_FAMILY_STATUS')
onehotEncode(df,'NAME_EDUCATION_TYPE')
df.drop(columns=['NAME_HOUSING_TYPE','NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_EDUCATION_TYPE'],inplace=True)
print(df)

#Use ensemble learning to predict and evaluate (cv: KFold, GradientBoosting method)
#target: 'AMT_INCOME_TOTAL',
target = df.columns.tolist()
target.remove('AMT_INCOME_TOTAL')
target.remove('ID')
feature = {'AMT_INCOME_TOTAL'}

FoldNum = 5
x = pd.DataFrame(df.loc[:,target],columns=target)
y = df[['AMT_INCOME_TOTAL']]
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size = 1/FoldNum, shuffle=True)

#RandomForest
begin = time.time()
param_grid = {'n_estimators':[50,100,200],'max_features':[1,10,20],'criterion':["mse","friedman_mse"],'max_depth':[1,10,100,200]}
RFR_model = sklearn.ensemble.RandomForestRegressor(max_samples = 20)

RFR_model.fit(x_train,y_train.values.ravel())
kfold = KFold(FoldNum,shuffle=True,random_state=1)
RFR_gscv = GridSearchCV(RFR_model, param_grid,cv=kfold,n_jobs=-1)
RFR_gscv.fit(x_train,y_train.values.ravel())

y_predict = RFR_gscv.best_estimator_.predict(x_test)
RFR_score = RFR_gscv.score(x_test,y_test)
RFR_mse = mean_squared_error(y_predict,y_test).round(2)

RFR_compareY = pd.DataFrame.copy(y_test)
RFR_compareY['predict'] = y_predict.round()
RFR_compareY['difference'] = abs(RFR_compareY['AMT_INCOME_TOTAL'] - RFR_compareY['predict'])
RFR_compareY['error rate'] = (RFR_compareY['difference']/RFR_compareY['AMT_INCOME_TOTAL']*100).round(2).astype(str)+"%"
end = time.time()
time_spent_RFR = round(end-begin,3)
print("\n",RFR_compareY)

#Bagging
begin = time.time()
best_estimator = [DecisionTreeRegressor(max_depth=1,criterion="mse"),DecisionTreeRegressor(max_depth=10,criterion="mse"),DecisionTreeRegressor(max_depth=100,criterion="mse"),DecisionTreeRegressor(max_depth=200,criterion="mse"),DecisionTreeRegressor(max_depth=1,criterion="friedman_mse"),DecisionTreeRegressor(max_depth=10,criterion="friedman_mse"),DecisionTreeRegressor(max_depth=100,criterion="friedman_mse"),DecisionTreeRegressor(max_depth=200,criterion="friedman_mse")]
param_grid = {'n_estimators':[100,200,300],'max_features':[1,4,7],'base_estimator':best_estimator}
Bagging_model = sklearn.ensemble.BaggingRegressor(max_samples = 20)

Bagging_model.fit(x_train,y_train.values.ravel())

kfold = KFold(5,shuffle=True,random_state=1)
Bagging_gscv = GridSearchCV(Bagging_model, param_grid,cv=kfold,n_jobs=-1)
Bagging_gscv.fit(x_train,y_train.values.ravel())

y_predict = Bagging_gscv.best_estimator_.predict(x_test)
Bagging_score = Bagging_gscv.score(x_test,y_test)
Bagging_mse = mean_squared_error(y_predict,y_test).round(2)

Bagging_compareY = pd.DataFrame.copy(y_test)
Bagging_compareY['predict'] = y_predict.round()
Bagging_compareY['difference'] = abs(Bagging_compareY['AMT_INCOME_TOTAL'] - Bagging_compareY['predict'])
Bagging_compareY['error rate'] = (Bagging_compareY['difference']/Bagging_compareY['AMT_INCOME_TOTAL']*100).round(2).astype(str)+"%"
end = time.time()
time_spent_Bagging = round(end-begin,3)
print("\n",Bagging_compareY)

#GradientBoost
begin = time.time()
param_grid = {'n_estimators':[50,100,200],'max_features':[1,10,20],'max_depth':[1,10,100,200],'criterion':["mse","friedman_mse"]}
GBR_model = sklearn.ensemble.GradientBoostingRegressor(learning_rate=0.01,min_samples_split=3)

kfold = KFold(FoldNum,shuffle=True,random_state=1)
GBR_gscv = GridSearchCV(GBR_model, param_grid,cv=kfold,n_jobs=-1)
GBR_gscv.fit(x_train,y_train.values.ravel())

y_predict = GBR_gscv.best_estimator_.predict(x_test)
GBR_score = GBR_gscv.score(x_test,y_test)
GBR_mse = mean_squared_error(y_predict,y_test).round(2)

GBR_compareY = pd.DataFrame.copy(y_test)
GBR_compareY['predict'] = y_predict.round()
GBR_compareY['difference'] = abs(GBR_compareY['AMT_INCOME_TOTAL'] - GBR_compareY['predict'])
GBR_compareY['error rate'] = (GBR_compareY['difference']/GBR_compareY['AMT_INCOME_TOTAL']*100).round(2).astype(str)+"%"
end = time.time()
time_spent_GBR = round(end-begin,3)
print("\n",GBR_compareY)

type = ['Random Forest','Bagging','Gradient Boosting']
score = [RFR_score,Bagging_score,GBR_score]
mse = [RFR_mse,Bagging_mse,GBR_mse]
time_spent = [time_spent_RFR,time_spent_Bagging,time_spent_GBR]
resultValue = {'type':type,'score':score,'mse':mse,'time_spent':time_spent}
result = pd.DataFrame(resultValue)

print(result)
