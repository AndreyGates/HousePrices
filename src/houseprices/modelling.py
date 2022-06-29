from houseprices.data_prep import data_extract, data_preprocessor
from houseprices.of_transformer import ordinal_transformer

from sklearn.ensemble import GradientBoostingRegressor

import csv


def gdboost_fitting(X_train, y_train):
    '''FITTING THE MODEL WITH TRAIN.CSV DATA AND RETURNING SCORE'''
    reg = GradientBoostingRegressor(n_estimators=500, max_leaf_nodes=20, learning_rate=0.061)
    reg.fit(X_train, y_train)
    return reg


def gdboost_testing(preprocessor, reg):
    '''PREDICTING ON TEST.CSV DATA'''
    X_test, y_test = data_extract('../HousePrices/tests/test.csv')
    X_test = ordinal_transformer(X_test)
    X_test = preprocessor.transform(X_test)

    y_test = reg.predict(X_test)

    return y_test


def fill_csv(y_test):
    '''FILLING IN CSV FOR SUBMISSION'''
    header = ['Id', 'SalePrice']
    data = zip(range(1461, 2920), y_test)

    with open('../HousePrices/tests/predictions.csv', 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(data)


def modelling():
    '''PREPROCESSING AND TRAIN-TEST SPLIT FOR TRAIN.CSV'''
    X_train, y_train = data_extract('../HousePrices/tests/train.csv')
    X_train = ordinal_transformer(X_train)
    preprocessor = data_preprocessor()
    X_train = preprocessor.fit_transform(X_train)

    '''CREATING GDBOOST REGRESSOR AND PREDICTING'''
    gdboost_reg = gdboost_fitting(X_train, y_train)
    y_test = gdboost_testing(preprocessor, gdboost_reg)
    fill_csv(y_test)

    return X_train, y_train, gdboost_reg  # to test its accuracy on training ds
