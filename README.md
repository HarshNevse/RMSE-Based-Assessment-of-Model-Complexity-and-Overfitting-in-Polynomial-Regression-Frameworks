# RMSE-Based Assessment of Model Complexity and Overfitting in Polynomial Regression Frameworks

## Overview
This project demonstrates overfitting in machine learning using polynomial regression on the **Advertising dataset**. By progressively increasing the degree of polynomial features, we analyze the bias-variance trade-off and how model complexity impacts training and testing performance.

## Key Features
- Implementation of **Polynomial Regression** with varying degrees of complexity.
- Quantitative analysis using **Root Mean Squared Error (RMSE)** as the evaluation metric.
- Visualization of training and testing RMSE trends to highlight overfitting behavior.

## Dataset
 ![{27E7E6FB-13E4-4259-8C5F-709683564C08}](https://github.com/user-attachments/assets/7aec8517-63f6-4aeb-9915-a7ede2efeced)

The dataset used is `Advertising.csv`, which contains the following features:
- **TV**: Advertising spend on TV (in thousands of dollars).
- **Radio**: Advertising spend on radio (in thousands of dollars).
- **Newspaper**: Advertising spend on newspapers (in thousands of dollars).
Label:
- **Sales**: Product sales (in thousands of units).

## Requirements
This project uses Python and the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Code Description
### Data Preprocessing
- The dataset is loaded and separated into predictors (`X`) and target variable (`y`).
- Polynomial features are generated for increasing degrees (1 to 9).

### Training and Testing
- Data is split into training (70%) and testing (30%) sets.
- A `LinearRegression` model is fit on the training data for each polynomial degree.

### Error Metrics
- The RMSE for both training and testing sets is calculated for each degree of polynomial features.
- Results are stored and visualized to show the impact of increasing model complexity.

### Visualization
1. **Scatter Plots**: Relationship between each predictor (TV, Newspaper, Radio) and the target variable (Sales).
   ![Untitled](https://github.com/user-attachments/assets/3172463e-2ce5-4a82-b6ef-8171204e4612)

3. **Regression Plot**: Comparison between predicted and actual sales for testing data.
   ![Untitled](https://github.com/user-attachments/assets/a27731f8-dbd5-4c8d-8b18-ce5f060dd26c)

5. **RMSE Trends**: Line plot showing train and test RMSE as a function of polynomial degree.
   ![Untitled](https://github.com/user-attachments/assets/8f295f1a-5059-4c18-9efb-92ea89e2dc2b)


## Results
The RMSE trends demonstrate:
- **Low-degree models (e.g., degree 1-3)**: Both train and test RMSE decrease, indicating underfitting.
- **Optimal degree (e.g., degree 4)**: Balanced performance on training and testing data.
- **High-degree models (e.g., degree 5-9)**: Training RMSE decreases further, but testing RMSE rises sharply, showing overfitting.

## Key Code Snippets
### RMSE Calculation Loop
```python
test_RMSE = []
train_RMSE = []

for i in range(1,10):
    
    ob2 = PolynomialFeatures(degree=i, include_bias=False)
    
    polymod2 = ob2.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(polymod2, y, test_size=0.3, random_state=101)
    
    ob3 = LinearRegression()
    mod3 = ob3.fit(X_train, y_train)
    
    ypredtest = mod3.predict(X_test)
    ypredtrain = mod3.predict(X_train)
    
    train_rmse = np.sqrt(mean_squared_error(y_train,ypredtrain))
    test_rmse = np.sqrt(mean_squared_error(y_test,ypredtest))
    
    train_RMSE.append(train_rmse)
    test_RMSE.append(test_rmse)
```

### RMSE Trend Visualization
```python
plt.figure(figsize=(10,6))
plt.plot(err['degree'][0:6],err['train_RMSE'][0:6])
plt.plot(err['degree'][0:6],err['test_RMSE'][0:6])
plt.legend(labels=['train_RMSE','test_RMSE'])
plt.xlabel('Dergree')
plt.ylabel('RMSE')
plt.ylim(0,4.6)
```

## Conclusion
This project successfully demonstrates how increasing polynomial degrees can lead to overfitting. By analyzing RMSE trends, we highlight the trade-off between bias and variance, providing valuable insights into model complexity and its impact on generalization.

## Future Work
- Introduce regularization techniques (e.g., Ridge or Lasso Regression) to mitigate overfitting.
- Use cross-validation to evaluate model performance more robustly.
- Explore feature selection and scaling to improve regression performance.
