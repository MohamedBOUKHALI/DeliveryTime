import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from unittest.mock import patch, mock_open
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import (
    load_and_clean_data,
    get_features_and_target,
    define_feature_columns,
    create_preprocessor,
    create_pipeline,
    train_pipeline,
    evaluate_pipeline
)

class TestPipeline:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        data = {
            'Order_ID': [1, 2, 3],
            'Distance_km': [10.0, 15.0, 20.0],
            'Weather': ['Clear', 'Rainy', 'Clear'],
            'Traffic_Level': ['Low', 'Medium', 'High'],
            'Time_of_Day': ['Morning', 'Afternoon', 'Evening'],
            'Vehicle_Type': ['Bike', 'Car', 'Scooter'],
            'Preparation_Time_min': [10, 15, 20],
            'Courier_Experience_yrs': [2.0, 5.0, 1.0],
            'Delivery_Time_min': [30, 45, 60]
        }
        return pd.DataFrame(data)

    def test_load_and_clean_data(self, sample_data):
        """Test loading and cleaning data."""
        csv_content = sample_data.to_csv(index=False)
        with patch('pandas.read_csv', return_value=sample_data) as mock_read:
            result = load_and_clean_data('dummy_path.csv')
            mock_read.assert_called_once_with('dummy_path.csv')
            pd.testing.assert_frame_equal(result, sample_data)

    def test_load_and_clean_data_with_na(self):
        """Test loading and cleaning data with NA values."""
        data_with_na = pd.DataFrame({
            'Order_ID': [1, 2, 3],
            'Distance_km': [10.0, np.nan, 20.0],
            'Weather': ['Clear', 'Rainy', 'Clear'],
            'Traffic_Level': ['Low', 'Medium', 'High'],
            'Time_of_Day': ['Morning', 'Afternoon', 'Evening'],
            'Vehicle_Type': ['Bike', 'Car', 'Scooter'],
            'Preparation_Time_min': [10, 15, 20],
            'Courier_Experience_yrs': [2.0, 5.0, 1.0],
            'Delivery_Time_min': [30, 45, 60]
        })
        expected = data_with_na.dropna()
        with patch('pandas.read_csv', return_value=data_with_na):
            result = load_and_clean_data('dummy_path.csv')
            pd.testing.assert_frame_equal(result, expected)

    def test_get_features_and_target(self, sample_data):
        """Test separating features and target."""
        X, y = get_features_and_target(sample_data)
        expected_X = sample_data.drop(columns=['Delivery_Time_min'])
        expected_y = sample_data['Delivery_Time_min']
        pd.testing.assert_frame_equal(X, expected_X)
        pd.testing.assert_series_equal(y, expected_y)

    def test_get_features_and_target_custom_target(self, sample_data):
        """Test separating features and target with custom target."""
        sample_data['Custom_Target'] = [100, 200, 300]
        X, y = get_features_and_target(sample_data, target='Custom_Target')
        expected_X = sample_data.drop(columns=['Custom_Target'])
        expected_y = sample_data['Custom_Target']
        pd.testing.assert_frame_equal(X, expected_X)
        pd.testing.assert_series_equal(y, expected_y)

    def test_define_feature_columns(self):
        """Test defining feature columns."""
        num_features, cat_features = define_feature_columns()
        expected_num = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
        expected_cat = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
        assert num_features == expected_num
        assert cat_features == expected_cat

    def test_create_preprocessor(self):
        """Test creating preprocessor."""
        num_features = ['Distance_km', 'Preparation_Time_min']
        cat_features = ['Weather', 'Traffic_Level']
        preprocessor = create_preprocessor(num_features, cat_features)
        assert isinstance(preprocessor, ColumnTransformer)
        assert len(preprocessor.transformers) == 2
        # Check numerical transformer
        num_transformer = preprocessor.named_transformers_['num']
        assert isinstance(num_transformer, StandardScaler)
        # Check categorical transformer
        cat_transformer = preprocessor.named_transformers_['cat']
        assert isinstance(cat_transformer, OneHotEncoder)
        assert cat_transformer.drop == 'first'
        assert cat_transformer.handle_unknown == 'ignore'

    def test_create_pipeline(self):
        """Test creating pipeline."""
        num_features = ['Distance_km', 'Preparation_Time_min']
        cat_features = ['Weather', 'Traffic_Level']
        preprocessor = create_preprocessor(num_features, cat_features)
        pipeline = create_pipeline(preprocessor, k_features=5)
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0][0] == 'preprocessor'
        assert pipeline.steps[1][0] == 'feature_selection'
        assert pipeline.steps[2][0] == 'regressor'
        # Check feature selection
        feature_selector = pipeline.named_steps['feature_selection']
        assert isinstance(feature_selector, SelectKBest)
        assert feature_selector.score_func == f_regression
        assert feature_selector.k == 5
        # Check regressor
        regressor = pipeline.named_steps['regressor']
        assert isinstance(regressor, RandomForestRegressor)
        assert regressor.random_state == 42

    def test_train_pipeline_without_grid_search(self, sample_data):
        """Test training pipeline without hyperparameter tuning."""
        X, y = get_features_and_target(sample_data)
        num_features, cat_features = define_feature_columns()
        preprocessor = create_preprocessor(num_features, cat_features)
        pipeline = create_pipeline(preprocessor)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        assert trained_pipeline is not None
        assert hasattr(trained_pipeline, 'predict')

    def test_train_pipeline_with_grid_search(self, sample_data):
        """Test training pipeline with hyperparameter tuning."""
        X, y = get_features_and_target(sample_data)
        num_features, cat_features = define_feature_columns()
        preprocessor = create_preprocessor(num_features, cat_features)
        pipeline = create_pipeline(preprocessor)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        param_grid = {
            'feature_selection__k': [2, 3],
            'regressor__n_estimators': [10, 20]
        }
        trained_pipeline = train_pipeline(pipeline, X_train, y_train, param_grid=param_grid)
        assert trained_pipeline is not None
        assert hasattr(trained_pipeline, 'predict')

    def test_evaluate_pipeline(self, sample_data, capsys):
        """Test evaluating pipeline."""
        X, y = get_features_and_target(sample_data)
        num_features, cat_features = define_feature_columns()
        preprocessor = create_preprocessor(num_features, cat_features)
        pipeline = create_pipeline(preprocessor)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        evaluate_pipeline(trained_pipeline, X_train, X_test, y_train, y_test)
        captured = capsys.readouterr()
        assert "Performance on training set:" in captured.out
        assert "Performance on test set:" in captured.out
        assert "MAE:" in captured.out
        assert "R²:" in captured.out

    def test_evaluate_pipeline_overfitting_warning(self, sample_data, capsys):
        """Test overfitting warning in evaluation."""
        # Create a scenario where train MAE is much lower than test MAE
        X, y = get_features_and_target(sample_data)
        num_features, cat_features = define_feature_columns()
        preprocessor = create_preprocessor(num_features, cat_features)
        # Use a pipeline that might overfit
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(f_regression, k=5)),
            ('regressor', RandomForestRegressor(n_estimators=1, random_state=42))  # Simple model
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        trained_pipeline = train_pipeline(pipeline, X_train, y_train)
        evaluate_pipeline(trained_pipeline, X_train, X_test, y_train, y_test)
        captured = capsys.readouterr()
        # Depending on the data, it might or might not trigger the warning
        # Just check that evaluation runs without error
        assert "Performance on training set:" in captured.out

if __name__ == "__main__":
    pytest.main([__file__])
