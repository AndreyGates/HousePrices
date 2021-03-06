import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector


def no_string_cols(df):
    '''REMOVING STRING COLUMNS'''
    cols_to_remove = []

    for col in df.columns:
        try:
            _ = df[col].astype(float)
        except ValueError:
            print('Couldn\'t convert %s to float' % col)
            cols_to_remove.append(col)
            pass

    # keep only the columns in df that do not contain string
    df = df[[col for col in df.columns if col not in cols_to_remove]]
    return df


def data_extract(csv_path):
    '''DATA EXTRACTION'''
    df = pd.read_csv(csv_path)

    # df = df.dropna(axis=1, thresh=len(df.values)/1.5) # dropping columns where 1/3 is Nan or more)
    # df = df.fillna(df.mean()) # imputing missing numerical values with feature means
    # df = df.drop(1379) # df[1379]['Electrical'] is the only empty string in the column

    X = df.drop(['Id'], axis=1)
    # X = X.drop(ordinal_columns_str, axis=1)
    y = None

    if csv_path == '../HousePrices/tests/train.csv':
        X = X.drop(['SalePrice'], axis=1)
        y = df['SalePrice']

    return X, y


def data_preprocessor():
    '''NUMERIC AND CATEGORICAL VALUES HANDLING (EXCEPT ORDINAL - DONE SEPARATELY)'''
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # ("pca", PCA(n_components=6))
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

    return preprocessor
