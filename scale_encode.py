import pandas as pd
import numpy as np
from sklearn import preprocessing

#onehot encoding
def onehotEncode(df, name):
   le = preprocessing.OneHotEncoder(handle_unknown='ignore')
   enc = df[[name]]
   enc = le.fit_transform(enc).toarray()
   enc_df = pd.DataFrame(enc, columns=le.categories_[0])
   df.loc[:, le.categories_[0]] = enc_df
   df.drop(columns=[name], inplace=True)

#label encoding
def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels

# get 2d array of dataframe with given dataframe
def get_various_encode_scale(dataframe, numerical_columns, categorical_columns):

    scale_Sd = preprocessing.StandardScaler()
    scale_Mm = preprocessing.MinMaxScaler()
    scale_Rb = preprocessing.RobustScaler()

    encode_label = preprocessing.LabelEncoder

    after_scale_encode = [[],[],[]]
    group_dataframe = []
    i = 0
    j = 0
    k = 0
    tmp_result = dataframe.copy

    while(k<3):
        after_scale_encode[k].append(dataframe.copy())
        after_scale_encode[k].append(dataframe.copy())
        k+=1

    for name in numerical_columns:

       scaled_Sd = scale_Sd.fit_transform(dataframe[name].values.reshape(-1,1))
       scaled_Mm = scale_Mm.fit_transform(dataframe[name].values.reshape(-1,1))
       scaled_Rb = scale_Rb.fit_transform(dataframe[name].values.reshape(-1,1))

       k=0
       while(k<2):
          after_scale_encode[0][k][name] = scaled_Sd
          after_scale_encode[1][k][name] = scaled_Mm
          after_scale_encode[2][k][name] = scaled_Rb
          k += 1

    for new in categorical_columns:
        k = 0
        while(k<3):
           labelEncode(after_scale_encode[k][0], new)
           onehotEncode(after_scale_encode[k][1], new)
           k+=1
    return after_scale_encode



