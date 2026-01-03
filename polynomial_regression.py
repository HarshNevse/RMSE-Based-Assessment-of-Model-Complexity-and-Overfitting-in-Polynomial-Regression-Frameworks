"""
Polynomial Regression Analysis Module

This module provides utilities for analyzing polynomial regression models
and assessing model complexity and overfitting using RMSE metrics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class PolynomialRegressionAnalyzer:
    """
    A class for performing polynomial regression analysis with varying degrees
    and evaluating overfitting using RMSE metrics.
    """
    
    def __init__(self, random_state=101):
        """
        Initialize the PolynomialRegressionAnalyzer.
        
        Parameters:
        -----------
        random_state : int, default=101
            Random state for reproducibility in train-test split
        """
        self.random_state = random_state
        self.models = {}
        self.results = None
        
    def fit_polynomial_models(self, X, y, max_degree=9, test_size=0.3):
        """
        Fit polynomial regression models for degrees 1 to max_degree.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        y : array-like or Series
            Target variable
        max_degree : int, default=9
            Maximum polynomial degree to test
        test_size : float, default=0.3
            Proportion of data to use for testing
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing degree, train_RMSE, and test_RMSE
        """
        if max_degree < 1:
            raise ValueError("max_degree must be at least 1")
        
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be array-like or DataFrame")
            
        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError("y must be array-like or Series")
            
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        test_RMSE = []
        train_RMSE = []
        
        for degree in range(1, max_degree + 1):
            # Generate polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y, test_size=test_size, random_state=self.random_state
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            # Calculate RMSE
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            train_RMSE.append(train_rmse)
            test_RMSE.append(test_rmse)
            
            # Store model
            self.models[degree] = {
                'model': model,
                'poly_features': poly_features
            }
        
        # Create results DataFrame
        self.results = pd.DataFrame({
            'degree': list(range(1, max_degree + 1)),
            'train_RMSE': train_RMSE,
            'test_RMSE': test_RMSE
        })
        
        return self.results
    
    def predict(self, X, degree):
        """
        Make predictions using a trained model of a specific degree.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix
        degree : int
            Polynomial degree of the model to use
            
        Returns:
        --------
        array
            Predicted values
        """
        if degree not in self.models:
            raise ValueError(f"Model with degree {degree} has not been trained")
        
        model_data = self.models[degree]
        X_poly = model_data['poly_features'].transform(X)
        return model_data['model'].predict(X_poly)
    
    def get_results(self):
        """
        Get the results DataFrame containing RMSE values for each degree.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with degree, train_RMSE, and test_RMSE columns
        """
        if self.results is None:
            raise ValueError("No models have been fitted yet. Call fit_polynomial_models first.")
        return self.results
    
    def get_optimal_degree(self):
        """
        Get the polynomial degree with the lowest test RMSE.
        
        Returns:
        --------
        int
            Optimal polynomial degree
        """
        if self.results is None:
            raise ValueError("No models have been fitted yet. Call fit_polynomial_models first.")
        
        min_idx = self.results['test_RMSE'].idxmin()
        return self.results.loc[min_idx, 'degree']


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_advertising_data(filepath):
    """
    Load the Advertising dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    tuple
        (X, y) where X is the feature matrix and y is the target variable
    """
    df = pd.read_csv(filepath)
    X = df.drop('sales', axis=1)
    y = df['sales']
    return X, y
