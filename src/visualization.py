import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from data_prep import *
from of_transformer import ordinal_transformer

def feature_selector(X, y, df):
    reg = RandomForestRegressor(n_estimators = 100, max_leaf_nodes=100)
    reg.fit(X, y)

    feat_labels = df.columns[:]
    importances = reg.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    plt.title('Feature importance')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), feat_labels[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()

df, y = data_extract('../HousePrices/src/train.csv')
X = ordinal_transformer(df)
preprocessor = data_preprocessor()
X = preprocessor.fit_transform(X)

feature_selector(X, y, df)

