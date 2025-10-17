import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def load_and_clean_data(filepath='../data/dataset.csv'):
    """
    Load the dataset and perform basic cleaning: drop missing values.
    """
    data = pd.read_csv(filepath)
    cleaned_data = data.dropna(axis=0).copy()
    return cleaned_data

def get_features_and_target(data, target='Delivery_Time_min'):
    """
    Separate features and target.
    """
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def define_feature_columns():
    """
    Define numerical and categorical feature columns.
    """
    numerical_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
    categorical_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    return numerical_features, categorical_features

def create_preprocessor(numerical_features, categorical_features):
    """
    Create the preprocessing ColumnTransformer.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor

def create_pipeline(preprocessor, k_features=10):
    """
    Create the full pipeline with preprocessing, feature selection, and regressor.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_regression, k=k_features)),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    return pipeline

def train_pipeline(pipeline, X_train, y_train, param_grid=None):
    """
    Train the pipeline, optionally with hyperparameter tuning using GridSearchCV.
    """
    if param_grid:
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline

def evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test):
    """
    Evaluate the pipeline on train and test sets.
    """
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("Performance on training set:")
    print(f"  - MAE: {train_mae:.2f} minutes")
    print(f"  - R²: {train_r2:.4f}")

    print("Performance on test set:")
    print(f"  - MAE: {test_mae:.2f} minutes")
    print(f"  - R²: {test_r2:.4f}")

    if train_mae < test_mae * 0.8:
        print("Warning: The model might be overfitting.")
    else:
        print("The model generalizes well.")

# Example usage (can be run as a script)
if __name__ == "__main__":
    # Load and clean data
    data = load_and_clean_data()
    X, y = get_features_and_target(data)

    # Define features
    num_features, cat_features = define_feature_columns()

    # Create preprocessor and pipeline
    preprocessor = create_preprocessor(num_features, cat_features)
    pipeline = create_pipeline(preprocessor)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define param grid for tuning (optional)
    param_grid = {
        'feature_selection__k': [10, 15, 20],
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }

    # Train pipeline with tuning
    trained_pipeline = train_pipeline(pipeline, X_train, y_train, param_grid=param_grid)

    # Evaluate
    evaluate_pipeline(trained_pipeline, X_train, X_test, y_train, y_test)
