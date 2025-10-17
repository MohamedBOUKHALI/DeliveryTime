import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def cleanData():
    data = pd.read_csv('dataset.csv')
    data = data.drop_duplicates()  
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    return data

def chooseXy(data):
   # Separate features and target
    X=data.drop(columns=['Delivery_Time_min','Order_ID','Courier_Experience_yrs'])
    y=data['Delivery_Time_min']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

#pipeline
def piplelineFunction(grid_params,model):
    data=cleanData()
    X_train, X_test, y_train, y_test=chooseXy(data)
    num_cols=['Distance_km','Preparation_Time_min']
    cat_cols=['Weather','Traffic_Level','Time_of_Day','Vehicle_Type']

    preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),num_cols),
        ('col',OneHotEncoder(handle_unknown='ignore'),cat_cols),
    ]
    )
    pipeline=Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',model)
    ])
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    #  Train GridSearch
    grid_search.fit(X_train, y_train)
    #  Evaluate on test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    return test_mae