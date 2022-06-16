from data_prep import data_extract, data_prep

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import csv


'''FITTING THE MODEL WITH TRAIN.CSV DATA (RF, ADABOOST, GDBOOST, XGBOOST) AND RETURNING SCORE'''
def rf(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(n_estimators = 100, max_leaf_nodes=100)
    reg.fit(X_train, y_train)
    return reg

def gdboost(X_train, X_test, y_train, y_test):
    reg = GradientBoostingRegressor(n_estimators = 200, max_leaf_nodes=20, learning_rate=0.2)
    reg.fit(X_train, y_train)
    return reg

'''PREDICTING ON TEST.CSV DATA'''
def model_application(X_train, y_train, preprocessor, reg):
    reg.fit(X_train, y_train)

    X_test, y_test = data_extract('../HousePrices/src/test.csv')
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
    X, preprocessor, y = data_prep('../HousePrices/src/train.csv')
    #X = ordinal_tranformer(X)
    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    '''CREATING RF AND GDBOOST REGRESSORS AND COMBINING THEM'''
    rf_reg = rf(X_train, X_test, y_train, y_test) 
    gdboost_reg = gdboost(X_train, X_test, y_train, y_test)

    y_test_rf = model_application(X, y, preprocessor, rf_reg)
    y_test_gdboost = model_application(X, y, preprocessor, gdboost_reg)

    y_test = (y_test_rf + y_test_gdboost) / 2
    fill_csv(y_test)

main()

# Next time continue hypertuning the RF regressor, visualize the error tendency based on the hyperparameters
# Also study gradient boosting and try to apply it too
# Then compare the results from both ensemble models
