from data_prep import data_extract, data_preprocessor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from of_transformer import ordinal_transformer

import csv

'''FITTING THE MODEL WITH TRAIN.CSV DATA AND RETURNING SCORE'''
def gdboost_fitting(X_train, y_train):
    reg = GradientBoostingRegressor(n_estimators = 200, max_leaf_nodes=100, learning_rate=0.15)
    reg.fit(X_train, y_train)
    return reg

'''PREDICTING ON TEST.CSV DATA'''
def gdboost_testing(preprocessor, reg):
    X_test, y_test = data_extract('../HousePrices/src/test.csv')
    X_test = ordinal_transformer(X_test)
    X_test = preprocessor.transform(X_test)

    y_test = reg.predict(X_test)

    return y_test

'''FILLING IN CSV FOR SUBMISSION'''
def fill_csv(y_test):
    header = ['Id', 'SalePrice']
    data = zip(range(1461, 2920), y_test)

    with open('../HousePrices/src/predictions.csv', 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(data)

def main():
    '''TRAIN-TEST SPLIT FOR TRAIN.CSV'''
    X_train, y_train = data_extract('../HousePrices/src/train.csv')
    X_train = ordinal_transformer(X_train)
    preprocessor = data_preprocessor()
    X_train = preprocessor.fit_transform(X_train)

    '''CREATING GDBOOST REGRESSOR AND PREDICTING'''
    gdboost_reg = gdboost_fitting(X_train, y_train)
    y_test = gdboost_testing(preprocessor, gdboost_reg)
    fill_csv(y_test)

main()

# Next time continue hypertuning the RF regressor, visualize the error tendency based on the hyperparameters
# Also study gradient boosting and try to apply it too
# Then compare the results from both ensemble models