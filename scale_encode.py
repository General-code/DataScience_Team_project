import pandas as pd
import numpy as np
from sklearn import preprocessing
"""
#onehot encoding
def onehotEncode(df, name):
   le = preprocessing.OneHotEncoder(handle_unknown='ignore')
   enc = df[[name]]
   enc = le.fit_transform(enc).toarray()
   enc_df = pd.DataFrame(enc, columns=le.categories_[0])
   df.loc[:, le.categories_[0]] = enc_df

def get_various_encode_scale(df, numerical_columns, categorical_columns):

    scale_Sd = preprocessing.StandardScaler
    scale_Mm = preprocessing.MinMaxScaler
    scale_Rb = preprocessing.RobustScaler


    encode_label = preprocessing.LabelEncoder

    after_scale_encode = []
    group_dataframe = []
    index = 0


    tmp_result = df

    after_scale_encode[0][numerical_columns] = scale_Sd.fit_transform(df[numerical_columns])
    after_scale_encode[1][numerical_columns] = scale_Mm.fit_transform(df[numerical_columns])
    after_scale_encode[2][numerical_columns] = scale_Rb.fit_transform(df[numerical_columns])





"""