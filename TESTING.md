# Testing Documentation

## Overview
This document describes the unit testing framework for the Polynomial Regression Analysis project.

## Test Structure

The test suite is organized into the following test classes:

### 1. TestPolynomialRegressionAnalyzer
Tests for the main `PolynomialRegressionAnalyzer` class:
- **Initialization tests**: Verify proper initialization with default and custom parameters
- **Model fitting tests**: Test `fit_polynomial_models()` with various inputs and parameters
- **Prediction tests**: Test `predict()` method for different polynomial degrees
- **Results tests**: Test `get_results()` and `get_optimal_degree()` methods
- **Edge case tests**: Invalid inputs, mismatched data lengths, invalid parameters
- **Reproducibility tests**: Verify consistent results with same random state

### 2. TestCalculateRMSE
Tests for the `calculate_rmse()` utility function:
- Perfect predictions (zero RMSE)
- Predictions with errors
- Various error magnitudes
- Different input types (arrays, lists)

### 3. TestLoadAdvertisingData
Tests for the `load_advertising_data()` function:
- Basic functionality with sample CSV
- Correct data loading and separation
- Error handling for non-existent files

### 4. TestIntegration
Integration tests covering complete workflows:
- Full workflow from data generation to prediction
- Overfitting detection with increasing polynomial degrees
- Small dataset handling

## Running Tests

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
pytest test_polynomial_regression.py -v
```

### Run Specific Test Class
```bash
pytest test_polynomial_regression.py::TestPolynomialRegressionAnalyzer -v
```

### Run Specific Test
```bash
pytest test_polynomial_regression.py::TestPolynomialRegressionAnalyzer::test_fit_polynomial_models_basic -v
```

### Run Tests with Coverage
```bash
pip install pytest-cov
pytest test_polynomial_regression.py --cov=polynomial_regression --cov-report=html
```

## Test Coverage

The test suite provides comprehensive coverage of:
- ✅ Core functionality (model fitting, prediction, RMSE calculation)
- ✅ Input validation and error handling
- ✅ Edge cases and boundary conditions
- ✅ Data type compatibility (numpy arrays, pandas DataFrames/Series, lists)
- ✅ Reproducibility and random state handling
- ✅ Integration tests for complete workflows

## Test Statistics

- **Total Tests**: 32
- **Test Classes**: 4
- **Test Methods**: 32
- **Expected Result**: All tests pass

## Continuous Integration

Tests can be integrated into CI/CD pipelines using:

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest test_polynomial_regression.py -v
```

## Key Test Scenarios

### 1. Valid Input Tests
- Various data sizes and shapes
- Different polynomial degrees (1-9)
- Different test/train split ratios
- Both numpy arrays and pandas DataFrames

### 2. Invalid Input Tests
- Invalid polynomial degrees (≤0)
- Mismatched X and y lengths
- Invalid test_size values
- Wrong data types
- Missing or non-existent files

### 3. Behavior Tests
- Models are properly stored after fitting
- Predictions are consistent for same input
- Optimal degree is correctly identified
- Results are reproducible with same random state

### 4. Edge Cases
- Small datasets (n < 10)
- Single feature datasets
- Perfect predictions (zero error)
- Various error magnitudes
