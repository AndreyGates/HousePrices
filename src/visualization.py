from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from data_prep import *
from of_transformer import ordinal_transformer

from matplotlib import pyplot as plt
from scipy.stats import loguniform

def l_curve(X_train, y_train, model):
    train_sizes, train_scores, test_scores = learning_curve(estimator=model, scoring='neg_mean_absolute_error', 
                                                           X=X_train, y=y_train,
                                                           train_sizes=np.linspace(0.1, 1.0, 10),
                                                           cv=10, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training neg_mae')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation neg_mae')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training examples')
    plt.ylabel('Neg_mae')
    plt.legend(loc='lower right')
    plt.ylim([-30000, 0])
    plt.show()

def hypertuning(X_train, y_train, model):
    param_distributions = {
    "n_estimators": [10, 20, 50, 100, 200, 500],
    "max_leaf_nodes": [10, 20, 50, 100],
    "learning_rate": loguniform(0.01, 1),
}
    cv = RandomizedSearchCV(estimator=model, 
                            param_distributions=param_distributions,
                            scoring='neg_mean_absolute_error', 
                            n_iter=20, 
                            random_state=0, 
                            cv=2,
                            n_jobs=-1)
    cv.fit(X_train, y_train)
    return cv

def main_visual():
    df, y = data_extract('../HousePrices/src/train.csv')
    X = ordinal_transformer(df)

    '''PIPELINING'''
    preprocessor = data_preprocessor()
    X = preprocessor.fit_transform(X)
    gboost = GradientBoostingRegressor()
    X_train, y_train = X, y

    '''HYPERTUNING'''
    rs = hypertuning(X_train, y_train, gboost) # using gridsearchCV to extract the best hyperparameters for the model
    scores = cross_val_score(rs, X_train, y_train, scoring='neg_mean_absolute_error', cv=5) # nested cv
    model = rs.best_estimator_
    print(rs.best_params_)
    l_curve(X_train, y_train, model) # looks fine
    
    print(f'CV MSE: {np.mean(scores):.3f} 'f'+/- {np.std(scores):.3f}')

main_visual()