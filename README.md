# Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML](https://img.shields.io/badge/Machine%20Learning-Classification-brightgreen)](https://scikit-learn.org/)

A robust Machine Learning system for detecting fraudulent credit card transactions. 
This project implements and compares four classification models (Logistic Regression, 
Decision Tree, Random Forest, and SVM) to identify potentially fraudulent activities 
with high accuracy. The system includes data preprocessing, feature engineering, 
and comprehensive model evaluation.

## 🚀 Features

- **Data Preprocessing**: Comprehensive cleaning and transformation of transaction data
- **Feature Engineering**: Creation of meaningful features for better fraud detection
- **Class Imbalance Handling**: Implementation of SMOTE for balanced model training
- **Multiple ML Models**: Comparison of various algorithms including:
  - Random Forest
  - Decision Tree
  - Logistic Regression
  - Support Vector Machine
- The model currently being used for predictions is: RandomForestClassifier(This model was chosen as the best performing model during training based on the F1-score)
- **Model Persistence**: Save and load trained models for production use
- **Comprehensive Evaluation**: Detailed metrics and visualizations for model assessment

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Project Structure

```
credit-card-fraud-detection/
├── data/                    # Dataset directory
│   └── fraud_7k_14p28.csv   # Sample dataset (not included in repo)
├── models/                  # Trained models and scalers
│   ├── best_fraud_model.pkl
│   └── robust_scaler.pkl
├── reports/                 # Generated reports and visualizations
│   ├── figures/             # Plots and charts
│   │   └── confusion_matrix.png
│   └── model_metrics.csv    # Performance metrics
├── fraud_detection.py       # Main application code
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## 🚦 Usage

1. **Prepare your data**:
   - Place your transaction data in CSV format in the `data/` directory
   - Ensure the data follows the expected format (refer to `data/README.md` for details)

2. **Run the training pipeline**:
   ```bash
   python fraud_detection.py
   ```
   This will:
   - Load and preprocess the data
   - Train multiple machine learning models
   - Evaluate and compare their performance
   - Save the best model to `models/`
   - Generate reports in the `reports/` directory

## 🔍 Making Predictions

Use the trained model to detect fraudulent transactions:

```python
from fraud_detection import predict_transaction

# Example transaction features: [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
sample_transaction = [1, 1000.0, 10000.0, 9000.0, 5000.0, 6000.0]
prediction = predict_transaction(sample_transaction)

print(f"Transaction ID: {sample_transaction[0]}")
print(f"Amount: ${sample_transaction[1]:.2f}")
print(f"Fraud Prediction: {'✅ Fraud Detected' if prediction['is_fraud'] else '✅ Legitimate'}")
print(f"Confidence: {prediction['fraud_probability']:.2%}")
```

## 📊 Model Evaluation

The system provides comprehensive evaluation metrics:

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 99.8%   |
| Precision | 0.92    |
| Recall    | 0.85    |
| F1-Score  | 0.88    |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  Output Screenshots
<img width="1912" height="593" alt="image" src="https://github.com/user-attachments/assets/ea741d59-59f0-4b9f-9027-8f3520bc495e" />
<img width="1919" height="983" alt="image" src="https://github.com/user-attachments/assets/69893465-8ce8-4eba-b441-ae8fba587d8a" />
<img width="1897" height="1024" alt="image" src="https://github.com/user-attachments/assets/a95e937e-4415-441b-b7ed-705499f72ef6" />
<img width="1889" height="955" alt="image" src="https://github.com/user-attachments/assets/69183482-cb31-4043-9a18-6035a5a23aee" />

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Built with ❤️ using Python and scikit-learn

---

<div align="center">
  Made with ❤️ by Priyanka Jakkampudi
</div>
