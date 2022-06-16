import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector


ordinal_columns_str = ['Utilities', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 
                       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                       'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 
                       'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                       'PoolQC', 'Fence', 'MiscFeature']

def data_extract(csv_path):
    df = pd.read_csv(csv_path)

    #df = df.dropna(axis=1, thresh=len(df.values)/1.5) # dropping columns where 1/3 is Nan or more)
    #df = df.fillna(df.mean()) # imputing missing numerical values with feature means
    #df = df.drop(1379) # df[1379]['Electrical'] is the only empty string in the column
    
    X = df.drop(['Id'], axis=1)
    X = X.drop(ordinal_columns_str, axis=1)
    y = None
    
    if csv_path == '../HousePrices/src/train.csv':
        X = X.drop(['SalePrice'], axis=1)
        y = df['SalePrice']

    return X, y


def data_prep(csv_path):
    '''HANDLING MISSING NUMERIC VALUES AND IMPUTING THEM'''
    df = pd.read_csv(csv_path)

    #df = df.dropna(axis=1, thresh=len(df.values)/1.5) # dropping columns where 1/3 is Nan or more)
    #df = df.fillna(df.mean()) # imputing missing numerical values with feature means
    #df = df.drop(1379) # df[1379]['Electrical'] is the only empty string in the column

    X, y = data_extract(csv_path)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    '''HANDLING CATEGORICAL DATA'''
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(categories='auto', drop='first'))
    ])

    '''OVERALL PREPROCESSING - COLUMN TRANSFORMATION'''
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=object)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ])

    return X, preprocessor, y

    #model = pipeline.fit(X_train, y_train)
    #print(model.score(X_train, y_train))
