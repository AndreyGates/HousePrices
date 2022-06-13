from data_prep import data_extract, data_prep

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

'''FITTING THE MODEL WITH TRAIN.CSV DATA'''
def model_fitting():
    X, preprocessor, y = data_prep('../HousePrices/src/train.csv')
    X = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

    reg = RandomForestRegressor(max_depth=20)
    
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    return reg

def model_evaluation():
    pass

def model_output():
    X_train, preprocessor, y_train = data_prep('../HousePrices/src/train.csv')
    X_test, y_test = data_extract('../HousePrices/src/test.csv')

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.fit_transform(X_test)

    reg = RandomForestRegressor(max_depth=20)

    reg.fit(X_train, y_train)
    y_test = reg.predict(X_test)

    return y_test

def main():
    y_test = model_output()

    print(y_test)
main()

# Next time continue hypertuning the RF regressor, visualize the error tendency based on the hyperparameters
# Also study gradient boosting and try to apply it too
# Then compare the results from both ensemble models