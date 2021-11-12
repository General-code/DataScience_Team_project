import numpy as np
import pandas as pd
from sklearn import preprocessing


def getEncode(df, name, encoder):
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels


# onehot Encoding
def onehotEncode(df, name):
    le = preprocessing.OneHotEncoder(handle_unknown='ignore')
    enc = df[[name]]
    enc = le.fit_transform(enc).toarray()
    le.categories_[0] = le.categories_[0].astype(np.str)
    new = np.full((len(le.categories_[0]), 1), name + ": ")
    le.categories_[0] = np.core.defchararray.add(new, le.categories_[0])
    enc_df = pd.DataFrame(enc, columns=le.categories_[0][0])
    df.reset_index(drop=True,inplace=True)
    df = pd.concat([df, enc_df], axis=1)
    df.drop(columns=[name], inplace=True)
    return df


# label encoding
def labelEncode(df, name):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df[name])
    labels = encoder.transform(df[name])
    df.loc[:, name] = labels




"""
Function to get 2d array of dataframe with given dataframe
output: scale/encoded 2d array dataframe, scalers used, encoders used
"""
def get_various_encode_scale(X, numerical_columns, categorical_columns, scalers=None, encoders=None, scaler_name=None,
                             encoder_name=None):
    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []

    if len(categorical_columns) == 0:
        return get_various_scale(X, numerical_columns, scalers, scaler_name)
    if len(numerical_columns) == 0:
        return get_various_encode(X, categorical_columns, encoders, encoder_name)

    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(), preprocessing.OneHotEncoder()]

    after_scale_encode = [[0 for col in range(len(encoders))] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale_encode[i].pop()
        for encode in encoders:
            after_scale_encode[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                after_scale_encode[i][j][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
                j = j + 1
            i = i + 1

    for new in categorical_columns:
        i = 0
        for scaler in scalers:
            j = 0
            for encoder in encoders:
                if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                    labelEncode(after_scale_encode[i][j], new)
                elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                    after_scale_encode[i][j] = onehotEncode(after_scale_encode[i][j], new)
                else:
                    getEncode(after_scale_encode[i][j], new, encoder)
                j = j + 1
            i = i + 1

    return after_scale_encode, scalers, encoders


"""
If there aren't categorical value, do this function
This function only scales given X
Output: 1d array of scaled dataset, scalers used, encoders used(Nothing)
"""
def get_various_scale(X, numerical_columns, scalers=None, scaler_name=None):
    """
    Test scale/encoder sets
    """
    if scalers is None:
        scalers = [preprocessing.StandardScaler(), preprocessing.MinMaxScaler(), preprocessing.RobustScaler()]
        # scalers = [preprocessing.StandardScaler()]
    encoders = ["None"]

    after_scale = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_scale[i].pop()
        for encode in encoders:
            after_scale[i].append(X.copy())
        i = i + 1

    for name in numerical_columns:
        i = 0
        for scaler in scalers:
            after_scale[i][0][name] = scaler.fit_transform(X[name].values.reshape(-1, 1))
            i = i + 1

    return after_scale, scalers, ["None"]


"""
If there aren't numerical value, do this function
This function only encodes given X
Return: 1d array of encoded dataset, scalers used(Nothing), encoders used
"""
def get_various_encode(X, categorical_columns, encoders=None, encoder_name=None):
    """
    Test scale/encoder sets
    """
    if encoders is None:
        encoders = [preprocessing.LabelEncoder(),preprocessing.OneHotEncoder()]
        # encoders = [preprocessing.LabelEncoder()]
    scalers = ["None"]

    after_encode = [[0 for col in range(1)] for row in range(len(scalers))]

    i = 0
    for scale in scalers:
        for encode in encoders:
            after_encode[i].pop()
        for encode in encoders:
            after_encode[i].append(X.copy())
        i = i + 1

    for new in categorical_columns:
        j = 0
        for encoder in encoders:
            if (str(type(encoder)) == "<class 'sklearn.preprocessing._label.LabelEncoder'>"):
                labelEncode(after_encode[0][j], new)
            elif (str(type(encoder)) == "<class 'sklearn.preprocessing._encoders.OneHotEncoder'>"):
                after_encode[0][j] = onehotEncode(after_encode[0][j], new)
            else:
                getEncode(after_encode[0][j], new, encoder)
            j = j + 1

    return after_encode, ["None"], encoders

