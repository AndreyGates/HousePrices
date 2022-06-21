from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from data_prep import *
from of_transformer import ordinal_transformer

from matplotlib import pyplot as plt
from scipy.stats import loguniform

def fitting(X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": 200,
        "max_leaf_nodes": 20,
        "learning_rate": 0.11,
        "random_state" : 69
    }

    reg = GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    mse = mean_squared_error(y_test, reg.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    return reg, params

def training_deviance(reg, params, X_test, y_test):
    test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()

def hypertuning(X_train, y_train):
    param_distributions = {
    "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
    "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
    "learning_rate": loguniform(0.01, 1),
}

    cv = RandomizedSearchCV(GradientBoostingRegressor(), 
                            param_distributions=param_distributions,
                            scoring="neg_mean_absolute_error", 
                            n_iter=20, 
                            random_state=0, 
                            n_jobs=2)
    cv.fit(X_train, y_train)
    return cv

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

def main_visual():
    df, y = data_extract('../HousePrices/src/train.csv')

    X = ordinal_transformer(df)
    preprocessor = data_preprocessor()
    X = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=69)

    #print(preprocessor.named_transformers_['num'][2].explained_variance_ratio_)
    display(hypertuning(X, y)) # using gridsearchCV to extract the best hyperparameters for the model

    #reg, params = fitting(X_train, y_train, X_test, y_test)
    #training_deviance(reg, params, X_test, y_test)

main_visual()