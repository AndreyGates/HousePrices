from data_prep import data_extract, data_prep

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


'''FITTING THE MODEL WITH TRAIN.CSV DATA (RF, ADABOOST, GDBOOST, XGBOOST)'''
def rf(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(n_estimators = 100, max_leaf_nodes=100)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    return reg

def gdboost(X_train, X_test, y_train, y_test):
    reg = GradientBoostingRegressor(n_estimators = 200, max_leaf_nodes=20, learning_rate=0.2)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    return reg


def model_application():
    X_train, preprocessor, y_train = data_prep('../HousePrices/src/train.csv')
    X_test, y_test = data_extract('../HousePrices/src/test.csv')

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.fit_transform(X_test)

    return y_test

def main():
    X, preprocessor, y = data_prep('../HousePrices/src/train.csv')
    X = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf(X_train, X_test, y_train, y_test)
    gdboost(X_train, X_test, y_train, y_test)

main()

# Next time continue hypertuning the RF regressor, visualize the error tendency based on the hyperparameters
# Also study gradient boosting and try to apply it too
# Then compare the results from both ensemble models