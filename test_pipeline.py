import pytest
from sklearn.svm import SVR
from pipeline import *
def test_cleanData():
    data=cleanData()
    assert isinstance(data,pd.DataFrame)
    assert data.isnull().sum().sum() == 0
    assert data.shape[0]==1000
    assert data.shape[1]==9

def test_piplelineFunction():
    grid_params = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }
    model = SVR()
    test_mae = piplelineFunction(grid_params, model)
    assert test_mae < 8