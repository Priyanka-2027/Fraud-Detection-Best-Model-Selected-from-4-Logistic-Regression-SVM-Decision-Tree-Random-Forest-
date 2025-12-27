# Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML](https://img.shields.io/badge/Machine%20Learning-Classification-brightgreen)](https://scikit-learn.org/)

A robust machine learning pipeline for detecting fraudulent credit card transactions using various classification algorithms. This system is specifically designed to handle the challenges of highly imbalanced datasets commonly encountered in fraud detection scenarios.

## 🚀 Features

- **Data Preprocessing**: Comprehensive cleaning and transformation of transaction data
- **Feature Engineering**: Creation of meaningful features for better fraud detection
- **Class Imbalance Handling**: Implementation of SMOTE for balanced model training
- **Multiple ML Models**: Comparison of various algorithms including:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - Support Vector Machine
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

## 🤝 Output Screenshots
<img width="1919" height="983" alt="image" src="https://github.com/user-attachments/assets/5f090c09-a650-44e7-b1c3-013421fbbd24" />
<img width="1911" height="928" alt="image" src="https://github.com/user-attachments/assets/bee81688-7d25-4ab0-966d-206417ed7848" />
<img width="1912" height="593" alt="image" src="https://github.com/user-attachments/assets/38b75450-a5d4-470f-841b-82c41a3207a0" />
<img width="298" height="11" alt="image" src="https://github.com/user-attachments/assets/5726d2ba-073e-4b96-b740-2e0081f18861" />

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- Built with ❤️ using Python and scikit-learn

---

<div align="center">
  Made with ❤️ by [Your Name] | [![GitHub](https://img.shields.io/github/followers/yourusername?style=social)](https://github.com/yourusername)
</div>
