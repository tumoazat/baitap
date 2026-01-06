"""
Model Training and Evaluation Module for Vietnam Housing Price Prediction

This module provides the HousingPriceModel class for training, evaluating,
and managing multiple machine learning models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import joblib
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')


class HousingPriceModel:
    """
    A comprehensive model trainer and evaluator for housing price prediction.
    
    This class handles:
    - Data preparation and splitting
    - Multiple model initialization
    - Model training and evaluation
    - Cross-validation
    - Hyperparameter tuning
    - Model persistence
    - Predictions
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: List[str] = []
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'Giá',
                    test_size: float = 0.2, drop_cols: Optional[List[str]] = None) -> Tuple:
        """
        Prepare data for training by splitting into train/test sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for testing
            drop_cols: Columns to drop before training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Drop specified columns
        if drop_cols is None:
            drop_cols = ['Ngày', 'Địa chỉ']
        
        df_model = df.copy()
        existing_drop_cols = [col for col in drop_cols if col in df_model.columns]
        if existing_drop_cols:
            df_model = df_model.drop(columns=existing_drop_cols)
        
        # Separate features and target
        X = df_model.drop(columns=[target_col])
        y = df_model[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"✓ Data prepared:")
        print(f"  - Training set: {len(self.X_train)} samples")
        print(f"  - Test set: {len(self.X_test)} samples")
        print(f"  - Features: {len(self.feature_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all machine learning models.
        
        Returns:
            Dictionary of initialized models
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        print(f"✓ Initialized {len(self.models)} models:")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        
        return self.models
    
    def train_model(self, model_name: str, X_train: Optional[pd.DataFrame] = None,
                   y_train: Optional[pd.Series] = None) -> Any:
        """
        Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features (uses stored if None)
            y_train: Training target (uses stored if None)
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Initialize models first.")
        
        X = X_train if X_train is not None else self.X_train
        y = y_train if y_train is not None else self.y_train
        
        if X is None or y is None:
            raise ValueError("Training data not available. Call prepare_data() first.")
        
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X, y)
        print(f"✓ {model_name} training completed")
        
        return model
    
    def evaluate_model(self, model_name: str, X_test: Optional[pd.DataFrame] = None,
                      y_test: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features (uses stored if None)
            y_test: Test target (uses stored if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        X = X_test if X_test is not None else self.X_test
        y = y_test if y_test is not None else self.y_test
        
        if X is None or y is None:
            raise ValueError("Test data not available. Call prepare_data() first.")
        
        model = self.models[model_name]
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Calculate MAPE
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100 if mask.any() else 0.0
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def cross_validate_model(self, model_name: str, cv: int = 5,
                            scoring: str = 'neg_mean_absolute_error') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name: Name of the model to validate
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            
        Returns:
            Dictionary with mean and std of CV scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Call prepare_data() first.")
        
        print(f"Cross-validating {model_name} with {cv} folds...")
        model = self.models[model_name]
        
        scores = cross_val_score(model, self.X_train, self.y_train,
                                cv=cv, scoring=scoring, n_jobs=-1)
        
        # Convert negative MAE back to positive
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        print(f"✓ CV Score: {cv_results['mean_score']:.2f} (+/- {cv_results['std_score']:.2f})")
        
        return cv_results
    
    def train_all_models(self, evaluate: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Train and optionally evaluate all models.
        
        Args:
            evaluate: Whether to evaluate models after training
            
        Returns:
            Dictionary of results for all models
        """
        if not self.models:
            self.initialize_models()
        
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60 + "\n")
        
        for model_name in self.models.keys():
            self.train_model(model_name)
            
            if evaluate:
                metrics = self.evaluate_model(model_name)
                print(f"\n{model_name} Results:")
                print(f"  MAE:  {metrics['MAE']:,.2f} VNĐ")
                print(f"  RMSE: {metrics['RMSE']:,.2f} VNĐ")
                print(f"  R²:   {metrics['R2']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
        
        # Find best model based on R² score
        if self.results:
            self.best_model_name = max(self.results, key=lambda k: self.results[k]['R2'])
            self.best_model = self.models[self.best_model_name]
            
            print("\n" + "="*60)
            print(f"BEST MODEL: {self.best_model_name}")
            print("="*60 + "\n")
        
        return self.results
    
    def hyperparameter_tuning(self, model_name: str, param_grid: Dict,
                             cv: int = 3, scoring: str = 'neg_mean_absolute_error') -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name: Name of the model to tune
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Best estimator from grid search
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not available. Call prepare_data() first.")
        
        print(f"\nTuning hyperparameters for {model_name}...")
        print(f"Parameter grid: {param_grid}")
        
        model = self.models[model_name]
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best score: {-grid_search.best_score_:.2f}")
        
        # Update model with best estimator
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Visualize model comparison results.
        
        Args:
            figsize: Figure size (width, height)
        """
        if not self.results:
            print("No results to plot. Train and evaluate models first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        models = list(self.results.keys())
        metrics_data = {
            'MAE': [self.results[m]['MAE'] for m in models],
            'RMSE': [self.results[m]['RMSE'] for m in models],
            'R2': [self.results[m]['R2'] for m in models],
            'MAPE': [self.results[m]['MAPE'] for m in models]
        }
        
        # Plot MAE
        axes[0, 0].bar(models, metrics_data['MAE'], color='skyblue')
        axes[0, 0].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('MAE (VNĐ)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot RMSE
        axes[0, 1].bar(models, metrics_data['RMSE'], color='lightcoral')
        axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('RMSE (VNĐ)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot R²
        axes[1, 0].bar(models, metrics_data['R2'], color='lightgreen')
        axes[1, 0].set_title('R² Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        # Plot MAPE
        axes[1, 1].bar(models, metrics_data['MAPE'], color='plum')
        axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, model_name: Optional[str] = None,
                             top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from a model.
        
        Args:
            model_name: Name of the model (uses best model if None)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        model_name = model_name or self.best_model_name
        
        if model_name is None:
            raise ValueError("No model specified and no best model found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_feature_importance(self, model_name: Optional[str] = None,
                               top_n: int = 10, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot feature importance.
        
        Args:
            model_name: Name of the model (uses best model if None)
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df.empty:
            return
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name or self.best_model_name}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_name: Optional[str] = None, filepath: str = 'model.pkl') -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save (uses best model if None)
            filepath: Path to save the model file
        """
        model_name = model_name or self.best_model_name
        
        if model_name is None:
            raise ValueError("No model specified and no best model found")
        
        model = self.models[model_name]
        
        # Save model and metadata
        model_data = {
            'model': model,
            'model_name': model_name,
            'feature_names': self.feature_names,
            'metrics': self.results.get(model_name, {})
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model
        """
        model_data = joblib.load(filepath)
        
        model_name = model_data['model_name']
        self.models[model_name] = model_data['model']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_name
        self.best_model = model_data['model']
        
        if 'metrics' in model_data:
            self.results[model_name] = model_data['metrics']
        
        print(f"✓ Model loaded from: {filepath}")
        print(f"  Model: {model_name}")
        print(f"  Features: {len(self.feature_names)}")
        
        return self.best_model
    
    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Input features
            model_name: Name of the model to use (uses best model if None)
            
        Returns:
            Array of predictions
        """
        model_name = model_name or self.best_model_name
        
        if model_name is None:
            raise ValueError("No model specified and no best model found")
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def predict_single(self, input_data: Dict[str, Any],
                      model_name: Optional[str] = None) -> float:
        """
        Make a single prediction from input dictionary.
        
        Args:
            input_data: Dictionary with feature values
            model_name: Name of the model to use
            
        Returns:
            Predicted price
        """
        # Create DataFrame with proper feature order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value
        
        # Reorder columns to match training data
        input_df = input_df[self.feature_names]
        
        prediction = self.predict(input_df, model_name)[0]
        
        return prediction
