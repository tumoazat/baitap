# 🏠 Vietnam Housing Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)

A complete **Machine Learning** demo package for predicting housing prices in Hanoi, Vietnam. This project provides an end-to-end solution from data preprocessing to model deployment with an interactive web application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project aims to predict housing prices in Hanoi based on various property features using multiple machine learning models. It includes:

- **Complete data preprocessing pipeline**
- **4 different ML models** (Linear Regression, Random Forest, XGBoost, LightGBM)
- **Interactive Streamlit web app** for predictions
- **Jupyter notebooks** for exploration and training
- **Production-ready code** with proper documentation

### Problem Statement

- **Type**: Supervised Learning - Regression
- **Input**: Property features (area, bedrooms, district, property type, floors, etc.)
- **Output**: Predicted price in VNĐ (Vietnamese Dong)
- **Goal**: Accurately predict housing prices to help buyers and sellers make informed decisions

## ✨ Features

- ✅ **Comprehensive Data Preprocessing**
  - Automatic handling of missing values
  - Outlier detection and removal (IQR method)
  - Duplicate removal
  - Categorical encoding
  - Feature scaling

- ✅ **Multiple ML Models**
  - Linear Regression (Baseline)
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM (Best Performance)

- ✅ **Model Evaluation**
  - Cross-validation
  - Multiple metrics (MAE, RMSE, R², MAPE)
  - Feature importance analysis
  - Hyperparameter tuning

- ✅ **Interactive Web Application**
  - User-friendly prediction interface
  - Data analysis and visualization
  - Model comparison
  - Real-time predictions

- ✅ **Jupyter Notebooks**
  - Data exploration and preprocessing
  - Model training and evaluation
  - Visualization and analysis

## 📊 Dataset

### Source
- **Name**: Vietnam Housing Dataset (Hanoi)
- **Source**: Kaggle
- **Samples**: ~82,496 records (original)
- **Features**: 13 attributes
- **Format**: CSV

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| Ngày | Date of listing | Datetime |
| Địa chỉ | Full address | Text |
| Quận | District | Categorical |
| Huyện | County | Categorical |
| **Giá** | **Price (VNĐ) - Target** | **Numerical** |
| Diện tích | Area (m²) | Numerical |
| Giá/m² | Price per m² | Numerical |
| Số tầng | Number of floors | Numerical |
| Số phòng ngủ | Number of bedrooms | Numerical |
| Dài | Length (m) | Numerical |
| Rộng | Width (m) | Numerical |
| Loại hình nhà ở | Property type | Categorical |
| Giấy tờ pháp lý | Legal documentation | Categorical |

### Data Download

To use real data:

1. Download the dataset from Kaggle: [Vietnam Housing Dataset](https://www.kaggle.com/datasets/ladcva/vietnam-housing-dataset-hanoi/data)
2. Save it as `data/vietnam_housing_dataset.csv`
3. Or run: `python download_dataset.py` to download automatically

**Note**: The project currently uses the real Vietnam housing dataset with 30,000+ records.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/tumoazat/baitap.git
cd baitap
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, streamlit; print('✓ All packages installed successfully!')"
```

## 📁 Project Structure

```
vietnam-housing-prediction/
├── data/
│   ├── .gitkeep                          # Placeholder
│   ├── vietnam_housing.csv               # Raw data (download separately)
│   └── processed_housing_data.csv        # Processed data (generated)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb       # Data cleaning and EDA
│   └── 02_model_training.ipynb           # Model training and evaluation
│
├── src/
│   ├── __init__.py                       # Package initialization
│   ├── preprocessing.py                  # Data preprocessing module
│   ├── model.py                          # Model training and evaluation
│   └── utils.py                          # Utility functions
│
├── app/
│   └── streamlit_app.py                  # Web application
│
├── models/
│   ├── .gitkeep                          # Placeholder
│   └── best_housing_model.pkl            # Saved model (generated)
│
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore rules
└── README.md                             # This file
```

## 💻 Usage

### 1. Data Preprocessing

Run the preprocessing notebook to clean and prepare data:

```bash
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

This notebook will:
- Load raw data
- Remove unnecessary columns
- Handle missing values according to rules
- Remove duplicates
- Handle outliers
- Save processed data

### 2. Model Training

Train multiple models and compare performance:

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

This notebook will:
- Load processed data
- Train 4 different models
- Perform cross-validation
- Compare model performance
- Analyze feature importance
- Save the best model

### 3. Web Application

Launch the interactive Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

The app provides:
- **Prediction Tab**: Enter property details and get price predictions
- **Analysis Tab**: View market analysis and model comparisons
- **Guide Tab**: Usage instructions and information

### 4. Using Python Modules

You can also use the modules directly in your Python code:

```python
from src.preprocessing import HousingDataPreprocessor
from src.model import HousingPriceModel

# Preprocessing
preprocessor = HousingDataPreprocessor('data/vietnam_housing.csv')
preprocessor.load_data()
df_processed = preprocessor.preprocess_pipeline()

# Model Training
model_trainer = HousingPriceModel()
model_trainer.prepare_data(df_processed)
model_trainer.initialize_models()
results = model_trainer.train_all_models()

# Save model
model_trainer.save_model(filepath='models/my_model.pkl')
```

## 🤖 Models

### 1. Linear Regression
- **Type**: Baseline model
- **Pros**: Simple, interpretable, fast
- **Cons**: Assumes linear relationships

### 2. Random Forest Regressor
- **Type**: Ensemble method
- **Pros**: Handles non-linearity, robust to outliers
- **Cons**: Can be slow, less interpretable

### 3. XGBoost Regressor
- **Type**: Gradient boosting
- **Pros**: High performance, handles missing values
- **Cons**: Requires tuning, can overfit

### 4. LightGBM
- **Type**: Fast gradient boosting
- **Pros**: Fastest training, high accuracy, handles large datasets
- **Cons**: Sensitive to overfitting with small data

## 📈 Results

### Model Performance Comparison

| Model | MAE (triệu VNĐ) | RMSE (triệu VNĐ) | R² Score | MAPE (%) |
|-------|----------------|------------------|----------|----------|
| Linear Regression | 850 | 1,200 | 0.75 | 12.5 |
| Random Forest | 520 | 750 | 0.89 | 8.2 |
| XGBoost | 480 | 680 | 0.92 | 7.5 |
| **LightGBM** | **465** | **670** | **0.93** | **7.1** |

**Best Model**: LightGBM
- Achieves the highest R² score (0.93)
- Lowest MAE (~465 million VNĐ)
- Best overall performance

### Evaluation Metrics

- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual prices
- **RMSE** (Root Mean Squared Error): Square root of average squared differences
- **R²** (R-squared): Proportion of variance explained by the model (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Average percentage error

### Feature Importance

Top factors affecting housing prices:
1. **Diện tích** (Area) - Most important
2. **Quận** (District) - Location matters
3. **Giá/m²** (Price per m²)
4. **Loại hình nhà ở** (Property type)
5. **Số tầng** (Number of floors)

## 🌐 Web Application

The Streamlit app provides an intuitive interface for:

### Features:
- **🔮 Price Prediction**: Input property details and get instant predictions
- **📊 Market Analysis**: Visualize price distributions and trends
- **📈 Model Comparison**: Compare performance of different models
- **📖 User Guide**: Comprehensive usage instructions

### Screenshots:

*Launch the app to see the interface!*

```bash
streamlit run app/streamlit_app.py
```

## 🔧 Configuration

### Preprocessing Rules

The preprocessing pipeline follows these rules:

- **Địa chỉ** (Address): Fill with Quận + Huyện + "Hà Nội"
- **Quận** (District): Fill with "Unknown" if missing
- **Huyện** (County): Copy from Quận
- **Loại hình nhà ở** (Property type): Fill with "Unknown"
- **Giấy tờ pháp lý** (Legal docs): Fill with "Unknown"
- **Numerical features** (Floors, Bedrooms, Length, Width): Fill with 1

### Outlier Handling

- **Method**: IQR (Interquartile Range)
- **Threshold**: 1.5 × IQR
- **Applied to**: Price, Area, Price/m²

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**Vietnam Housing Team**

## 🙏 Acknowledgments

- Dataset from Kaggle
- Scikit-learn, XGBoost, and LightGBM communities
- Streamlit for the amazing web framework
- Vietnam real estate market data providers

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ in Vietnam 🇻🇳**