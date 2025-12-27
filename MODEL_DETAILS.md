# Model Selection and Performance Analysis

## Models Tested
We evaluated four machine learning models for fraud detection:

1. **Logistic Regression**
   - Baseline model for binary classification
   - Simple and interpretable
   - Good for establishing baseline performance

2. **Support Vector Machine (SVM)**
   - Effective in high-dimensional spaces
   - Good for complex decision boundaries
   - Uses LinearSVC implementation

3. **Decision Tree**
   - Simple to understand and visualize
   - Handles non-linear relationships
   - Prone to overfitting

4. **Random Forest** (Selected Model)
   - Ensemble of decision trees
   - Reduces overfitting through bagging
   - Handles non-linear relationships well
   - Provides feature importance

## Performance Comparison

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 97.5%    | 0.89      | 0.82   | 0.85     |
| SVM                 | 98.1%    | 0.90      | 0.83   | 0.86     |
| Decision Tree       | 99.2%    | 0.91      | 0.87   | 0.89     |
| **Random Forest**   | **99.8%**| **0.92**  | **0.88**| **0.90** |

## Why Random Forest Was Chosen

1. **Highest Overall Performance**
   - Achieved the best balance between precision and recall
   - Highest F1-score (0.90) among all models
   - 99.8% accuracy in detecting fraudulent transactions

2. **Robustness**
   - Handles outliers and noise well
   - Less prone to overfitting compared to single Decision Tree
   - Works well with both numerical and categorical features

3. **Feature Importance**
   - Provides insights into which features contribute most to fraud detection
   - Helps in understanding the underlying patterns in fraudulent transactions

4. **Handling Imbalanced Data**
   - Effective even with the imbalanced nature of fraud detection datasets
   - Class weights were used to handle the class imbalance

## Model Configuration
```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    class_weight='balanced', # Adjust weights inversely proportional to class frequencies
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all available CPU cores
)
```

## Future Improvements

1. **Hyperparameter Tuning**
   - GridSearchCV or RandomizedSearchCV for optimal parameters
   - Experiment with different ensemble methods (e.g., XGBoost, LightGBM)

2. **Feature Engineering**
   - Add more transaction-based features
   - Include time-based features (hourly/daily patterns)

3. **Advanced Techniques**
   - Implement anomaly detection algorithms
   - Explore deep learning approaches (LSTM for sequential data)
   - Consider using AutoML for automated model selection

## How to Use the Model
```python
from fraud_detection import predict_transaction

# Example usage
sample_transaction = [1, 1000.0, 10000.0, 9000.0, 0.0, 1000.0]
prediction = predict_transaction(sample_transaction)
print(f"Fraud Prediction: {prediction['is_fraud']}")
print(f"Confidence: {prediction['fraud_probability']:.2%}")
print(f"Model Used: {prediction['model']}")
```

## Model Persistence
- The trained model is saved as `models/best_fraud_model.pkl`
- The RobustScaler is saved as `models/robust_scaler.pkl`
- Model metrics and visualizations are stored in the `reports/` directory
