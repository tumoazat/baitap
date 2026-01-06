"""
Data Preprocessing Module for Vietnam Housing Price Prediction

This module provides the HousingDataPreprocessor class for handling
data loading, cleaning, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


class HousingDataPreprocessor:
    """
    A comprehensive data preprocessor for Vietnam Housing dataset.
    
    This class handles all preprocessing steps including:
    - Loading data
    - Removing unnecessary columns
    - Handling duplicates
    - Managing missing values
    - Outlier detection and removal
    - Feature encoding
    - Feature scaling
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            data_path: Path to CSV file (overrides initialization path)
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        path = data_path or self.data_path
        
        if path is None:
            raise ValueError("Data path must be provided either during initialization or in load_data()")
        
        try:
            self.df = pd.read_csv(path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def remove_unnecessary_columns(self, columns_to_remove: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove unnecessary columns from the dataset.
        
        Args:
            columns_to_remove: List of column names to remove
            
        Returns:
            DataFrame with columns removed
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        # Default columns to remove
        if columns_to_remove is None:
            columns_to_remove = ['Unnamed: 0']
        
        # Filter to only existing columns
        existing_cols = [col for col in columns_to_remove if col in self.df.columns]
        
        if existing_cols:
            self.df = self.df.drop(columns=existing_cols)
            print(f"✓ Removed {len(existing_cols)} unnecessary column(s): {existing_cols}")
        else:
            print("✓ No unnecessary columns to remove")
        
        return self.df
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
            DataFrame with duplicates removed
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_count = initial_count - len(self.df)
        
        print(f"✓ Removed {removed_count} duplicate row(s)")
        return self.df
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values according to specific rules:
        - Địa chỉ: Fill with Quận + Huyện + "Hà Nội"
        - Quận: Fill with "Unknown" (only 2 records)
        - Huyện: Copy from Quận if available
        - Loại hình nhà ở, Giấy tờ pháp lý: Fill with "Unknown"
        - Số tầng, Số phòng ngủ, Dài, Rộng: Fill with 1
        
        Returns:
            DataFrame with missing values handled
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        print("Handling missing values...")
        
        # Địa chỉ: Combine Quận + Huyện + "Hà Nội"
        if 'Địa chỉ' in self.df.columns:
            mask = self.df['Địa chỉ'].isnull()
            if mask.any():
                quan = self.df.loc[mask, 'Quận'].fillna('')
                huyen = self.df.loc[mask, 'Huyện'].fillna('')
                self.df.loc[mask, 'Địa chỉ'] = quan + ', ' + huyen + ', Hà Nội'
                print(f"  - Filled {mask.sum()} missing 'Địa chỉ' values")
        
        # Quận: Fill with "Unknown"
        if 'Quận' in self.df.columns:
            missing_count = self.df['Quận'].isnull().sum()
            if missing_count > 0:
                self.df['Quận'].fillna('Unknown', inplace=True)
                print(f"  - Filled {missing_count} missing 'Quận' values with 'Unknown'")
        
        # Huyện: Copy from Quận
        if 'Huyện' in self.df.columns and 'Quận' in self.df.columns:
            mask = self.df['Huyện'].isnull()
            if mask.any():
                self.df.loc[mask, 'Huyện'] = self.df.loc[mask, 'Quận']
                print(f"  - Filled {mask.sum()} missing 'Huyện' values from 'Quận'")
        
        # Loại hình nhà ở: Fill with "Unknown"
        if 'Loại hình nhà ở' in self.df.columns:
            missing_count = self.df['Loại hình nhà ở'].isnull().sum()
            if missing_count > 0:
                self.df['Loại hình nhà ở'].fillna('Unknown', inplace=True)
                print(f"  - Filled {missing_count} missing 'Loại hình nhà ở' values")
        
        # Giấy tờ pháp lý: Fill with "Unknown"
        if 'Giấy tờ pháp lý' in self.df.columns:
            missing_count = self.df['Giấy tờ pháp lý'].isnull().sum()
            if missing_count > 0:
                self.df['Giấy tờ pháp lý'].fillna('Unknown', inplace=True)
                print(f"  - Filled {missing_count} missing 'Giấy tờ pháp lý' values")
        
        # Numerical columns: Fill with 1
        numerical_cols = ['Số tầng', 'Số phòng ngủ', 'Dài', 'Rộng']
        for col in numerical_cols:
            if col in self.df.columns:
                missing_count = self.df[col].isnull().sum()
                if missing_count > 0:
                    self.df[col].fillna(1, inplace=True)
                    print(f"  - Filled {missing_count} missing '{col}' values with 1")
        
        print("✓ Missing values handled successfully")
        return self.df
    
    def handle_outliers(self, columns: Optional[List[str]] = None, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers using IQR or Z-score method.
        
        Args:
            columns: List of column names to check for outliers (None for all numerical)
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier (default: 1.5) or Z-score threshold (default: 3)
            
        Returns:
            DataFrame with outliers removed
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        if columns is None:
            # Default to key numerical columns
            columns = ['Giá', 'Diện tích', 'Giá/m²']
            columns = [col for col in columns if col in self.df.columns]
        
        initial_count = len(self.df)
        
        if method == 'iqr':
            print(f"Removing outliers using IQR method (threshold={threshold})...")
            for col in columns:
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    print(f"  - Removed {outlier_count} outliers from '{col}'")
        
        elif method == 'zscore':
            print(f"Removing outliers using Z-score method (threshold={threshold})...")
            for col in columns:
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                    outlier_count = (z_scores > threshold).sum()
                    self.df = self.df[z_scores <= threshold]
                    print(f"  - Removed {outlier_count} outliers from '{col}'")
        
        removed_count = initial_count - len(self.df)
        print(f"✓ Total rows removed: {removed_count}")
        
        return self.df
    
    def encode_categorical(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            columns: List of categorical columns to encode (None for auto-detect)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        if columns is None:
            # Auto-detect categorical columns
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
            # Remove columns that shouldn't be encoded
            columns = [col for col in columns if col not in ['Ngày', 'Địa chỉ']]
        
        print("Encoding categorical variables...")
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  - Encoded '{col}' ({len(le.classes_)} unique values)")
        
        print("✓ Categorical encoding completed")
        return self.df
    
    def scale_features(self, columns: Optional[List[str]] = None,
                      exclude_target: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            columns: List of columns to scale (None for all numerical)
            exclude_target: Whether to exclude target variable 'Giá'
            
        Returns:
            DataFrame with scaled features
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        if columns is None:
            # Auto-detect numerical columns
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if exclude_target and 'Giá' in columns:
                columns.remove('Giá')
        
        if not columns:
            print("✓ No columns to scale")
            return self.df
        
        print("Scaling features...")
        self.scaler = StandardScaler()
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        
        print(f"✓ Scaled {len(columns)} feature(s)")
        return self.df
    
    def preprocess_pipeline(self, remove_outliers: bool = True,
                          encode_categorical: bool = True,
                          scale_features: bool = False) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            remove_outliers: Whether to remove outliers
            encode_categorical: Whether to encode categorical variables
            scale_features: Whether to scale numerical features
            
        Returns:
            Fully preprocessed DataFrame
        """
        print("\n" + "="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Remove unnecessary columns
        self.remove_unnecessary_columns()
        
        # Step 2: Remove duplicates
        self.remove_duplicates()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Handle outliers (optional)
        if remove_outliers:
            self.handle_outliers()
        
        # Step 5: Encode categorical variables (optional)
        if encode_categorical:
            self.encode_categorical()
        
        # Step 6: Scale features (optional)
        if scale_features:
            self.scale_features()
        
        self.df_processed = self.df.copy()
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETED")
        print("="*60)
        print(f"Final shape: {self.df_processed.shape[0]} rows × {self.df_processed.shape[1]} columns\n")
        
        return self.df_processed
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed DataFrame.
        
        Returns:
            Processed DataFrame
        """
        if self.df_processed is None:
            if self.df is not None:
                return self.df
            raise ValueError("No data available. Load and preprocess data first.")
        return self.df_processed
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed data to CSV file.
        
        Args:
            output_path: Path to save the processed CSV file
        """
        if self.df_processed is None:
            raise ValueError("No processed data to save. Run preprocess_pipeline() first.")
        
        self.df_processed.to_csv(output_path, index=False)
        print(f"✓ Processed data saved to: {output_path}")
