from houseprices.modelling import modelling


def test_accuracy():
    X_train, y_train, model = modelling()
    r2 = model.score(X_train, y_train)
    assert r2 >= 0 and r2 <= 1
