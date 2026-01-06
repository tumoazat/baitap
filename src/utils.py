"""
Utility Functions for Vietnam Housing Price Prediction

This module provides various utility functions for data formatting,
visualization, and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any


def format_price(price: float) -> str:
    """
    Format price in Vietnamese Dong (VNĐ).
    
    Args:
        price: Price value in VNĐ
        
    Returns:
        Formatted price string (e.g., "5,000,000,000 VNĐ")
    """
    try:
        return f"{price:,.0f} VNĐ"
    except (ValueError, TypeError):
        return "N/A"


def format_area(area: float) -> str:
    """
    Format area in square meters.
    
    Args:
        area: Area value in m²
        
    Returns:
        Formatted area string (e.g., "100.5 m²")
    """
    try:
        return f"{area:.1f} m²"
    except (ValueError, TypeError):
        return "N/A"


def calculate_price_per_sqm(price: float, area: float) -> float:
    """
    Calculate price per square meter.
    
    Args:
        price: Total price in VNĐ
        area: Area in m²
        
    Returns:
        Price per square meter
    """
    try:
        if area > 0:
            return price / area
        return 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def print_data_info(df: pd.DataFrame, title: str = "Dataset Information") -> None:
    """
    Print comprehensive information about the dataset.
    
    Args:
        df: Input DataFrame
        title: Title for the information display
    """
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    print("Column Information:")
    print("-" * 80)
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': [df[col].count() for col in df.columns],
        'Null Count': [df[col].isnull().sum() for col in df.columns],
        'Dtype': [df[col].dtype for col in df.columns]
    })
    print(info_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("=" * 80)


def plot_distribution(data: pd.Series, title: str = "Distribution Plot", 
                     xlabel: str = "Value", figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot distribution of a numerical feature.
    
    Args:
        data: Series containing numerical data
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Histogram with KDE
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=50)
    plt.title(f"{title} - Histogram")
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    
    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data)
    plt.title(f"{title} - Box Plot")
    plt.ylabel(xlabel)
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10),
                            title: str = "Correlation Matrix") -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input DataFrame with numerical columns
        figsize: Figure size (width, height)
        title: Plot title
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    if numerical_df.empty:
        print("No numerical columns found for correlation matrix.")
        return
    
    plt.figure(figsize=figsize)
    correlation = numerical_df.corr()
    
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(data: pd.Series, threshold: float = 1.5) -> Tuple[pd.Series, int]:
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Args:
        data: Series containing numerical data
        threshold: IQR multiplier (default: 1.5)
        
    Returns:
        Tuple of (boolean mask for outliers, count of outliers)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_count = outliers.sum()
    
    return outliers, outlier_count


def calculate_metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing MAE, RMSE, R², and MAPE
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def create_sample_input(district: str = "Ba Đình", property_type: str = "Nhà riêng",
                       area: float = 100.0, floors: int = 3, bedrooms: int = 3,
                       length: float = 10.0, width: float = 10.0,
                       legal_doc: str = "Sổ đỏ/ Sổ hồng") -> Dict[str, Any]:
    """
    Create a sample input dictionary for prediction.
    
    Args:
        district: District name
        property_type: Type of property
        area: Area in m²
        floors: Number of floors
        bedrooms: Number of bedrooms
        length: Length in meters
        width: Width in meters
        legal_doc: Legal documentation type
        
    Returns:
        Dictionary with sample input data
    """
    return {
        'Quận': district,
        'Loại hình nhà ở': property_type,
        'Diện tích': area,
        'Số tầng': floors,
        'Số phòng ngủ': bedrooms,
        'Dài': length,
        'Rộng': width,
        'Giấy tờ pháp lý': legal_doc
    }


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance") -> None:
    """
    Pretty print model performance metrics.
    
    Args:
        metrics: Dictionary containing metric names and values
        title: Title for the metrics display
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for metric_name, value in metrics.items():
        if metric_name == 'R2':
            print(f"{metric_name:.<25} {value:.4f}")
        elif metric_name == 'MAPE':
            print(f"{metric_name:.<25} {value:.2f}%")
        else:
            print(f"{metric_name:.<25} {format_price(value)}")
    
    print("=" * 60 + "\n")


def get_feature_importance_df(model: Any, feature_names: List[str], 
                              top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return (None for all)
        
    Returns:
        DataFrame with features and their importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute.")
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    return importance_df


def validate_input_data(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, str]:
    """
    Validate input data for prediction.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Check for None or empty values
    empty_fields = [field for field in required_fields 
                   if data[field] is None or data[field] == '']
    
    if empty_fields:
        return False, f"Empty values in fields: {', '.join(empty_fields)}"
    
    return True, ""
