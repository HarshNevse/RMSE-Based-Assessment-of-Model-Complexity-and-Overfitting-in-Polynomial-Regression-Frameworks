"""
Unit tests for the Polynomial Regression Analysis Module
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from polynomial_regression import (
    PolynomialRegressionAnalyzer,
    calculate_rmse,
    load_advertising_data
)


class TestPolynomialRegressionAnalyzer:
    """Test suite for PolynomialRegressionAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample regression data for testing"""
        X, y = make_regression(
            n_samples=100,
            n_features=3,
            n_informative=3,
            noise=10,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def analyzer(self):
        """Create a PolynomialRegressionAnalyzer instance"""
        return PolynomialRegressionAnalyzer(random_state=42)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.random_state == 42
        assert analyzer.models == {}
        assert analyzer.results is None
    
    def test_initialization_with_default(self):
        """Test analyzer initialization with default parameters"""
        analyzer = PolynomialRegressionAnalyzer()
        assert analyzer.random_state == 101
        assert analyzer.models == {}
        assert analyzer.results is None
    
    def test_fit_polynomial_models_basic(self, analyzer, sample_data):
        """Test basic functionality of fit_polynomial_models"""
        X, y = sample_data
        results = analyzer.fit_polynomial_models(X, y, max_degree=3)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert list(results.columns) == ['degree', 'train_RMSE', 'test_RMSE']
        assert list(results['degree']) == [1, 2, 3]
        assert all(results['train_RMSE'] > 0)
        assert all(results['test_RMSE'] > 0)
    
    def test_fit_polynomial_models_stores_models(self, analyzer, sample_data):
        """Test that fit_polynomial_models stores trained models"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=3)
        
        assert len(analyzer.models) == 3
        assert 1 in analyzer.models
        assert 2 in analyzer.models
        assert 3 in analyzer.models
        
        for degree in [1, 2, 3]:
            assert 'model' in analyzer.models[degree]
            assert 'poly_features' in analyzer.models[degree]
    
    def test_fit_polynomial_models_with_dataframe(self, analyzer):
        """Test fit_polynomial_models with pandas DataFrame"""
        df = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'feature3': np.random.rand(50)
        })
        y = pd.Series(np.random.rand(50))
        
        results = analyzer.fit_polynomial_models(df, y, max_degree=2)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
    
    def test_fit_polynomial_models_invalid_max_degree(self, analyzer, sample_data):
        """Test that fit_polynomial_models raises error for invalid max_degree"""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="max_degree must be at least 1"):
            analyzer.fit_polynomial_models(X, y, max_degree=0)
        
        with pytest.raises(ValueError, match="max_degree must be at least 1"):
            analyzer.fit_polynomial_models(X, y, max_degree=-1)
    
    def test_fit_polynomial_models_invalid_types(self, analyzer):
        """Test that fit_polynomial_models raises error for invalid input types"""
        with pytest.raises(TypeError, match="X must be array-like or DataFrame"):
            analyzer.fit_polynomial_models("invalid", np.array([1, 2, 3]))
        
        with pytest.raises(TypeError, match="y must be array-like or Series"):
            analyzer.fit_polynomial_models(np.array([[1, 2], [3, 4]]), "invalid")
    
    def test_fit_polynomial_models_mismatched_lengths(self, analyzer):
        """Test that fit_polynomial_models raises error for mismatched X and y lengths"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2])
        
        with pytest.raises(ValueError, match="X and y must have the same length"):
            analyzer.fit_polynomial_models(X, y)
    
    def test_fit_polynomial_models_invalid_test_size(self, analyzer, sample_data):
        """Test that fit_polynomial_models raises error for invalid test_size"""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            analyzer.fit_polynomial_models(X, y, test_size=0)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            analyzer.fit_polynomial_models(X, y, test_size=1)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            analyzer.fit_polynomial_models(X, y, test_size=-0.1)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            analyzer.fit_polynomial_models(X, y, test_size=1.5)
    
    def test_fit_polynomial_models_custom_test_size(self, analyzer, sample_data):
        """Test fit_polynomial_models with custom test_size"""
        X, y = sample_data
        results = analyzer.fit_polynomial_models(X, y, max_degree=2, test_size=0.2)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
    
    def test_predict_basic(self, analyzer, sample_data):
        """Test basic prediction functionality"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=2)
        
        predictions = analyzer.predict(X[:10], degree=1)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
    
    def test_predict_different_degrees(self, analyzer, sample_data):
        """Test prediction with different polynomial degrees"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=3)
        
        pred1 = analyzer.predict(X[:5], degree=1)
        pred2 = analyzer.predict(X[:5], degree=2)
        pred3 = analyzer.predict(X[:5], degree=3)
        
        assert len(pred1) == len(pred2) == len(pred3) == 5
        # Predictions from different models should generally be different
        assert not np.allclose(pred1, pred2) or not np.allclose(pred2, pred3)
    
    def test_predict_untrained_degree(self, analyzer, sample_data):
        """Test that predict raises error for untrained degree"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=2)
        
        with pytest.raises(ValueError, match="Model with degree 5 has not been trained"):
            analyzer.predict(X[:10], degree=5)
    
    def test_get_results_basic(self, analyzer, sample_data):
        """Test get_results method"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=3)
        
        results = analyzer.get_results()
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert list(results.columns) == ['degree', 'train_RMSE', 'test_RMSE']
    
    def test_get_results_before_fitting(self, analyzer):
        """Test that get_results raises error before fitting"""
        with pytest.raises(ValueError, match="No models have been fitted yet"):
            analyzer.get_results()
    
    def test_get_optimal_degree_basic(self, analyzer, sample_data):
        """Test get_optimal_degree method"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=5)
        
        optimal_degree = analyzer.get_optimal_degree()
        
        assert isinstance(optimal_degree, (int, np.integer))
        assert 1 <= optimal_degree <= 5
    
    def test_get_optimal_degree_before_fitting(self, analyzer):
        """Test that get_optimal_degree raises error before fitting"""
        with pytest.raises(ValueError, match="No models have been fitted yet"):
            analyzer.get_optimal_degree()
    
    def test_get_optimal_degree_correctness(self, analyzer, sample_data):
        """Test that get_optimal_degree returns the degree with minimum test RMSE"""
        X, y = sample_data
        analyzer.fit_polynomial_models(X, y, max_degree=4)
        
        optimal_degree = analyzer.get_optimal_degree()
        results = analyzer.get_results()
        
        min_rmse_idx = results['test_RMSE'].idxmin()
        expected_degree = results.loc[min_rmse_idx, 'degree']
        
        assert optimal_degree == expected_degree
    
    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random_state"""
        X, y = sample_data
        
        analyzer1 = PolynomialRegressionAnalyzer(random_state=123)
        results1 = analyzer1.fit_polynomial_models(X, y, max_degree=3)
        
        analyzer2 = PolynomialRegressionAnalyzer(random_state=123)
        results2 = analyzer2.fit_polynomial_models(X, y, max_degree=3)
        
        pd.testing.assert_frame_equal(results1, results2)
    
    def test_different_random_states(self, sample_data):
        """Test that different random states produce different results"""
        X, y = sample_data
        
        analyzer1 = PolynomialRegressionAnalyzer(random_state=123)
        results1 = analyzer1.fit_polynomial_models(X, y, max_degree=3)
        
        analyzer2 = PolynomialRegressionAnalyzer(random_state=456)
        results2 = analyzer2.fit_polynomial_models(X, y, max_degree=3)
        
        # Results should be different with different random states
        assert not results1.equals(results2)


class TestCalculateRMSE:
    """Test suite for calculate_rmse function"""
    
    def test_calculate_rmse_basic(self):
        """Test basic RMSE calculation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        assert rmse == 0.0
    
    def test_calculate_rmse_with_error(self):
        """Test RMSE calculation with prediction errors"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        # All predictions are off by 1, so RMSE should be 1
        assert np.isclose(rmse, 1.0)
    
    def test_calculate_rmse_large_error(self):
        """Test RMSE calculation with larger errors"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([3, 4, 0, 0])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        # MSE = (9 + 16 + 0 + 0) / 4 = 6.25, RMSE = 2.5
        assert np.isclose(rmse, 2.5)
    
    def test_calculate_rmse_positive_value(self):
        """Test that RMSE is always positive"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([3, 2, 1])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        assert rmse >= 0
    
    def test_calculate_rmse_with_lists(self):
        """Test RMSE calculation with list inputs"""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        
        rmse = calculate_rmse(y_true, y_pred)
        
        assert rmse == 0.0


class TestLoadAdvertisingData:
    """Test suite for load_advertising_data function"""
    
    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Create a sample CSV file for testing"""
        csv_content = """TV,radio,newspaper,sales
230.1,37.8,69.2,22.1
44.5,39.3,45.1,10.4
17.2,45.9,69.3,9.3
151.5,41.3,58.5,18.5
180.8,10.8,58.4,12.9
"""
        csv_file = tmp_path / "test_advertising.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)
    
    def test_load_advertising_data_basic(self, sample_csv_file):
        """Test basic functionality of load_advertising_data"""
        X, y = load_advertising_data(sample_csv_file)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 5
        assert len(y) == 5
        assert list(X.columns) == ['TV', 'radio', 'newspaper']
        assert y.name == 'sales'
    
    def test_load_advertising_data_values(self, sample_csv_file):
        """Test that load_advertising_data loads correct values"""
        X, y = load_advertising_data(sample_csv_file)
        
        # Check first row
        assert X.iloc[0]['TV'] == 230.1
        assert X.iloc[0]['radio'] == 37.8
        assert X.iloc[0]['newspaper'] == 69.2
        assert y.iloc[0] == 22.1
    
    def test_load_advertising_data_no_sales_in_X(self, sample_csv_file):
        """Test that 'sales' column is not in X"""
        X, y = load_advertising_data(sample_csv_file)
        
        assert 'sales' not in X.columns
    
    def test_load_advertising_data_file_not_found(self):
        """Test that load_advertising_data raises error for non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_advertising_data('non_existent_file.csv')


class TestIntegration:
    """Integration tests for the entire workflow"""
    
    def test_full_workflow(self):
        """Test complete workflow from data generation to getting optimal degree"""
        # Generate data
        X, y = make_regression(
            n_samples=100,
            n_features=3,
            n_informative=3,
            noise=10,
            random_state=42
        )
        
        # Initialize analyzer
        analyzer = PolynomialRegressionAnalyzer(random_state=42)
        
        # Fit models
        results = analyzer.fit_polynomial_models(X, y, max_degree=5)
        
        # Verify results
        assert len(results) == 5
        assert all(results['train_RMSE'] > 0)
        assert all(results['test_RMSE'] > 0)
        
        # Get optimal degree
        optimal_degree = analyzer.get_optimal_degree()
        assert 1 <= optimal_degree <= 5
        
        # Make predictions
        predictions = analyzer.predict(X[:10], degree=optimal_degree)
        assert len(predictions) == 10
        
        # Calculate RMSE of predictions
        rmse = calculate_rmse(y[:10], predictions)
        assert rmse >= 0
    
    def test_overfitting_detection(self):
        """Test that increasing polynomial degree eventually leads to overfitting"""
        # Generate simple data
        rng = np.random.default_rng(42)
        X = rng.random((50, 2))
        y = 2 * X[:, 0] + 3 * X[:, 1] + rng.standard_normal(50) * 0.1
        
        analyzer = PolynomialRegressionAnalyzer(random_state=42)
        results = analyzer.fit_polynomial_models(X, y, max_degree=8)
        
        # Training RMSE should generally decrease with complexity
        train_rmse_values = results['train_RMSE'].values
        # At least some progression toward lower training RMSE
        assert train_rmse_values[-1] <= train_rmse_values[0]
    
    def test_small_dataset(self):
        """Test analyzer with a small dataset"""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([1, 2, 3, 4, 5])
        
        analyzer = PolynomialRegressionAnalyzer(random_state=42)
        results = analyzer.fit_polynomial_models(X, y, max_degree=2, test_size=0.4)
        
        assert len(results) == 2
        assert all(results['train_RMSE'] >= 0)
        assert all(results['test_RMSE'] >= 0)
