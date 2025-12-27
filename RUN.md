# Fraud Detection System - Setup and Execution Guide

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## 1. Project Setup

### Option A: Using Git
```bash
# Clone the repository
git clone <repository-url>
cd MLL
```

### Option B: Using Downloaded Files
1. Extract the project folder
2. Open a terminal in the project directory

## 2. Environment Setup

### Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Running the Application

### Option 1: Web Interface (Recommended)
```bash
streamlit run app.py
```
- Access the interface at: `http://localhost:8501`
- Features:
  - Interactive dashboard
  - Real-time fraud predictions
  - Model performance metrics
  - Data visualizations

### Option 2: Command Line Interface
```bash
# Run the main script
python fraud_detection.py

# For prediction only
python -c "from fraud_detection import predict_transaction; print(predict_transaction([1, 1000.0, 10000.0, 9000.0, 0.0, 1000.0]))"
```

## 4. Project Structure
```
MLL/
├── data/                   # Dataset directory
│   └── fraud_7k_14p28.csv  # Sample dataset
├── models/                 # Trained models
├── reports/                # Evaluation reports and figures
├── app.py                 # Streamlit web interface
├── fraud_detection.py     # Core ML functionality
└── requirements.txt       # Dependencies
```

## 5. Making Predictions

### Using Python
```python
from fraud_detection import predict_transaction

transaction = [1, 1000.0, 10000.0, 9000.0, 0.0, 1000.0]  # Example transaction
result = predict_transaction(transaction)
print(f"Fraud: {result['is_fraud']} | Confidence: {result['fraud_probability']:.2%}")
```

## 6. Troubleshooting

### Common Issues
1. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Windows
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   
   # macOS/Linux
   lsof -i :8501
   kill -9 <PID>
   ```

3. **File Not Found**
   - Ensure all files are in the correct directories
   - Check file paths in the code

## 7. Support
For additional help, please contact [Your Contact Information]

---

*Last Updated: December 2025*
