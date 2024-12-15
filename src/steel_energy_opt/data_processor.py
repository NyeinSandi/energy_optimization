"""Data processing module for steel energy optimization."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the steel industry energy consumption dataset."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for model training."""
        # Convert date to datetime with European format (day first)
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
        
        # Create time-based features
        df['hour'] = df['date'].dt.hour
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Convert categorical variables
        df['Load_Type'] = pd.Categorical(df['Load_Type']).codes
        df['WeekStatus'] = (df['WeekStatus'] == 'Weekday').astype(int)  # Convert to binary
        
        return df
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        # Select features for training
        feature_columns = [
            'NSM', 'WeekStatus', 'day_of_week', 'Load_Type',
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'hour', 'month'
        ]
        
        X = df[feature_columns].values
        y = df['Usage_kWh'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, 
                   val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """Split data into train, validation, and test sets."""
        # Calculate split indices
        n_samples = len(X)
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split the data
        X_train = X[:val_idx]
        y_train = y[:val_idx]
        X_val = X[val_idx:test_idx]
        y_val = y[val_idx:test_idx]
        X_test = X[test_idx:]
        y_test = y[test_idx:]
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
