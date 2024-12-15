"""Neural network model for energy consumption prediction."""
import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional, Dict
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnergyPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """Initialize the energy prediction model."""
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        ])
        
        for _ in range(num_layers - 1):
            self.layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            
        self.layers.append(nn.Linear(hidden_size, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """Initialize the model trainer."""
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate various performance metrics."""
        # Convert to numpy for sklearn metrics
        outputs_np = outputs.detach().numpy().flatten()
        targets_np = targets.detach().numpy()
        
        # Calculate R² score
        r2 = r2_score(targets_np, outputs_np)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = mean_absolute_percentage_error(targets_np, outputs_np) * 100
        
        # Calculate MSE
        mse = self.criterion(outputs, targets.unsqueeze(1)).item()
        
        return {
            'r2_score': r2,
            'mape': mape,
            'mse': mse
        }
        
    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Perform one training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(X)
        loss = self.criterion(outputs, y.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Calculate metrics
        metrics = self.calculate_metrics(outputs, y)
        return metrics
        
    def validate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            metrics = self.calculate_metrics(outputs, y)
        return metrics

    def evaluate_model(self, X_test: torch.Tensor, y_test: torch.Tensor, verbose: bool = True) -> Dict[str, float]:
        """Evaluate model on test set and print detailed metrics."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            metrics = self.calculate_metrics(outputs, y_test)
            
            if verbose:
                print("\nTest Set Performance:")
                print(f"R² Score: {metrics['r2_score']:.4f}")
                print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
                print(f"Mean Squared Error: {metrics['mse']:.4f}")
                
            return metrics
        
    def train(self, 
              train_data: Tuple[torch.Tensor, torch.Tensor],
              val_data: Tuple[torch.Tensor, torch.Tensor],
              epochs: int = 100,
              early_stopping_patience: int = 10) -> dict:
        """Train the model with early stopping."""
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_mse': [], 'val_mse': [],
            'train_r2': [], 'val_r2': [],
            'train_mape': [], 'val_mape': []
        }
        
        # Use tqdm for progress bar
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            train_metrics = self.train_step(X_train, y_train)
            val_metrics = self.validate(X_val, y_val)
            
            # Store metrics
            history['train_mse'].append(train_metrics['mse'])
            history['val_mse'].append(val_metrics['mse'])
            history['train_r2'].append(train_metrics['r2_score'])
            history['val_r2'].append(val_metrics['r2_score'])
            history['train_mape'].append(train_metrics['mape'])
            history['val_mape'].append(val_metrics['mape'])
            
            # Update progress bar
            pbar.set_postfix({
                'train_mse': f'{train_metrics["mse"]:.4f}',
                'val_mse': f'{val_metrics["mse"]:.4f}',
                'train_r2': f'{train_metrics["r2_score"]:.4f}',
                'val_mape': f'{val_metrics["mape"]:.2f}%'
            })
            
            # Early stopping check
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience and val_metrics['r2_score'] < 0.95:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Print final metrics
        print("\nFinal Training Metrics:")
        print(f"Training R² Score: {history['train_r2'][-1]:.4f}")
        print(f"Validation R² Score: {history['val_r2'][-1]:.4f}")
        print(f"Training MAPE: {history['train_mape'][-1]:.2f}%")
        print(f"Validation MAPE: {history['val_mape'][-1]:.2f}%")
                
        return history
        
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions using the trained model."""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions
