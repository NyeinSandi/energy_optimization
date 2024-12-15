"""Training script for the energy optimization model."""
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from .data_processor import DataProcessor
from .model import EnergyPredictor, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_path: str, model_save_dir: str):
    """Train the energy optimization model."""
    # Initialize data processor
    processor = DataProcessor()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = processor.load_data(data_path)
    df = processor.preprocess_data(df)
    X, y = processor.prepare_features(df)
    
    # Split data
    data_splits = processor.split_data(X, y)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(data_splits['X_train'])
    y_train = torch.FloatTensor(data_splits['y_train'])
    X_val = torch.FloatTensor(data_splits['X_val'])
    y_val = torch.FloatTensor(data_splits['y_val'])
    X_test = torch.FloatTensor(data_splits['X_test'])
    y_test = torch.FloatTensor(data_splits['y_test'])
    
    # Initialize model
    input_size = X_train.shape[1]
    model = EnergyPredictor(input_size=input_size)
    trainer = ModelTrainer(model)
    
    # Train model
    logger.info("Starting model training...")
    history = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        epochs=10000
    )
    
    # Evaluate on test set
    test_loss = trainer.validate(X_test, y_test)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(model_save_dir) / f"model_{timestamp}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'history': history,
        'test_loss': test_loss
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")
    
if __name__ == "__main__":
    data_path = "data/raw/Steel_industry_data.csv"
    model_save_dir = "models"
    train_model(data_path, model_save_dir)
