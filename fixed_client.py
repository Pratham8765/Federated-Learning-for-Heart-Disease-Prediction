# fixed_client.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import flwr as fl
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging
import time
import sys
import argparse
import socket
import torch.utils.data as data
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fixed_client.log")
    ]
)
logger = logging.getLogger(__name__)

# Tabular Data Augmentation for Diabetes Dataset
class TabularDataAugmentation:
    """Data augmentation techniques for tabular medical data"""
    
    def __init__(self, noise_std=0.1, feature_noise_ratio=0.3):
        self.noise_std = noise_std
        self.feature_noise_ratio = feature_noise_ratio
    
    def add_gaussian_noise(self, X):
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, self.noise_std, X.shape)
        return X + noise
    
    def feature_dropout(self, X):
        """Randomly set some features to zero (simulate missing values)"""
        mask = np.random.random(X.shape) > self.feature_noise_ratio
        return X * mask
    
    def mixup(self, X, y, alpha=0.2):
        """Mixup augmentation - blend samples"""
        if X.shape[0] < 2:
            return X, y
            
        lam = np.random.beta(alpha, alpha)
        batch_size = X.shape[0]
        index = np.random.permutation(batch_size)
        
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y
    
    def augment_batch(self, X, y):
        """Apply multiple augmentation techniques"""
        # Apply noise to 50% of samples
        if np.random.random() > 0.5:
            X = self.add_gaussian_noise(X)
        
        # Apply feature dropout to 30% of samples
        if np.random.random() > 0.7:
            X = self.feature_dropout(X)
        
        # Apply mixup to 40% of samples
        if np.random.random() > 0.6 and X.shape[0] > 1:
            X, y = self.mixup(X, y)
        
        return X, y

# Load global scaler
def load_global_scaler():
    """Load the global scaler for consistent feature scaling"""
    try:
        scaler = joblib.load('global_scaler.pkl')
        logger.info("✅ Global scaler loaded successfully")
        return scaler
    except FileNotFoundError:
        logger.error("❌ Global scaler not found. Run setup_scaler.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading global scaler: {e}")
        return None

def load_diabetes_data(file_path='diabetes_non_negative_part1_2000.csv', test_size=0.2, client_id=1):
    """Load and preprocess the Diabetes dataset with client-specific splits"""
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        logger.info(f"Dataset shape: {df.shape}")
        
        # Define expected features
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Check if all expected features are present
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing expected features: {missing_features}")
            X = df.iloc[:, :8].values
        else:
            X = df[feature_columns].values
            
        # Target is the 'Outcome' column if it exists
        if 'Outcome' in df.columns:
            y = df['Outcome'].values
        else:
            y = df.iloc[:, -1].values
        
        # Convert target to binary if needed
        if len(np.unique(y)) > 2:
            y = (y > 0).astype(int)
        
        logger.info(f"Loaded {X.shape[1]} features and {y.shape[0]} samples")
        
        # Create Non-IID split for different clients
        if client_id == 1:
            # Client 1: First 60% of data (younger patients, lower glucose)
            split_idx = int(0.6 * len(X))
            X_client = X[:split_idx]
            y_client = y[:split_idx]
        else:
            # Client 2: Last 40% of data (older patients, higher glucose)
            split_idx = int(0.6 * len(X))
            X_client = X[split_idx:]
            y_client = y[split_idx:]
        
        logger.info(f"Client {client_id} received {len(X_client)} samples")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=test_size, random_state=42 + client_id, stratify=y_client
        )
        
        # Load GLOBAL scaler and apply consistent scaling
        global_scaler = load_global_scaler()
        if global_scaler is not None:
            X_train = global_scaler.transform(X_train)
            X_test = global_scaler.transform(X_test)
            logger.info("✅ Applied global scaler to training data")
        else:
            logger.error("❌ No global scaler available - aborting")
            raise FileNotFoundError("Global scaler not found. Run setup_scaler.py first.")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        return X_train, X_test, y_train, y_test, global_scaler
        
    except Exception as e:
        logger.error(f"Error loading the dataset: {e}")
        raise

# Simplified Model Architecture for Better Generalization
class ImprovedDiabetesModel(nn.Module):
    def __init__(self, input_size=8, num_classes=2):
        super(ImprovedDiabetesModel, self).__init__()
        # Simplified architecture to reduce overfitting
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 16),  # Reduced from 32
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.6)  # Increased dropout
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.5)  # Increased dropout
        )
        self.output = nn.Linear(8, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # Ensure input is 2D: [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output(x)

class FixedFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, client_id, global_scaler):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.client_id = client_id
        self.global_scaler = global_scaler
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Data augmentation
        self.augmenter = TabularDataAugmentation(noise_std=0.05, feature_noise_ratio=0.2)
        
        # Loss function with label smoothing
        self.label_smoothing = 0.1
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Optimizer with configurable hyperparameters
        self.learning_rate = 0.001
        self.weight_decay = 5e-4  # Increased regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7,  # More aggressive reduction
            patience=2,
            min_lr=1e-5
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 3
        self.no_improve_epochs = 0
        
        logger.info(f"✅ Fixed Client {client_id} initialized on device: {self.device}")
        logger.info(f"📏 Using global scaler: {self.global_scaler is not None}")
    
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        try:
            state_dict = self.model.state_dict()
            param_dict = {name: param for name, param in zip(state_dict.keys(), parameters)}
            
            for name, param in state_dict.items():
                if name in param_dict:
                    server_param = param_dict[name]
                    if isinstance(server_param, np.ndarray):
                        server_param = torch.from_numpy(server_param)
                    
                    if param.shape == server_param.shape:
                        state_dict[name] = server_param.to(self.device)
                    else:
                        logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {server_param.shape}")
            
            self.model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            logger.error(f"Error in set_parameters: {str(e)}")
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Update hyperparameters from server config
        epochs = config.get("epochs", 1)
        learning_rate = config.get("learning_rate", self.learning_rate)
        weight_decay = config.get("weight_decay", self.weight_decay)
        proximal_mu = config.get("proximal_mu", 0.1)
        
        # Update optimizer with new hyperparameters
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
            param_group['weight_decay'] = weight_decay
        
        # Save global parameters for FedProx
        if proximal_mu > 0:
            global_params = [p.detach().clone() for p in self.model.parameters()]
        
        # Training loop with data augmentation
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # Create augmented dataset for this epoch
            X_train_aug = self.X_train.numpy()
            y_train_aug = self.y_train.numpy()
            
            # Apply data augmentation
            X_train_aug, y_train_aug = self.augmenter.augment_batch(X_train_aug, y_train_aug)
            
            # Convert back to tensors
            X_train_aug = torch.FloatTensor(X_train_aug)
            y_train_aug = torch.LongTensor(y_train_aug)
            
            # Create data loader
            train_dataset = data.TensorDataset(X_train_aug, y_train_aug)
            train_loader = data.DataLoader(
                train_dataset, 
                batch_size=32, 
                shuffle=True,
                drop_last=True
            )
            
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # Standard cross-entropy loss
                loss = self.criterion(outputs, batch_y)
                
                # Add FedProx term if enabled
                if proximal_mu > 0:
                    proximal_term = 0.
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += (local_param - global_param).norm(2)
                    loss += (proximal_mu / 2) * proximal_term
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
            
            # Validation
            val_loss, val_metrics = self._evaluate_validation()
            self.scheduler.step(val_loss)
            
            # Log progress
            logger.info(f"Client {self.client_id} Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {epoch_loss/len(train_loader):.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        avg_loss = total_loss / (num_batches * epochs) if num_batches > 0 else 0
        
        return self.get_parameters(), len(self.X_train), {
            "loss": avg_loss,
            "client_id": self.client_id
        }
    
    def _evaluate_validation(self):
        """Helper method to evaluate on validation set"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            X_test_tensor = self.X_test.to(self.device)
            y_test_tensor = self.y_test.to(self.device)
            
            outputs = self.model(X_test_tensor)
            val_loss = self.criterion(outputs, y_test_tensor).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total = y_test_tensor.size(0)
            correct = (predicted == y_test_tensor).sum().item()
        
        self.model.train()
        return val_loss, {"accuracy": correct / total if total > 0 else 0.0}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            X_test_tensor = self.X_test.to(self.device)
            y_test_tensor = self.y_test.to(self.device)
            
            outputs = self.model(X_test_tensor)
            loss = self.criterion(outputs, y_test_tensor).item()
            
            # Calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test_tensor)
            
            y_true = y_test_tensor.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"Client {self.client_id} Evaluation - "
                       f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                       f"F1: {f1:.4f}")
        
        return float(loss), len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "client_id": self.client_id
        }

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP address: {e}")
        return "0.0.0.0"

def run_client():
    parser = argparse.ArgumentParser(description='Fixed Federated Learning Client for Diabetes Prediction')
    parser.add_argument('--server-address', type=str, default="127.0.0.1:8080",
                      help='Address of the server (default: 127.0.0.1:8080)')
    parser.add_argument('--client-id', type=int, required=True,
                      help='Unique identifier for this client')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--dataset', type=str, default='diabetes_non_negative_part1_2000.csv',
                      help='Path to the dataset file')
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 70)
        logger.info(f"🔧 FIXED Federated Learning Client (ID: {args.client_id})")
        logger.info("=" * 70)
        
        # Load and preprocess the dataset with Non-IID split and GLOBAL scaler
        X_train, X_test, y_train, y_test, global_scaler = load_diabetes_data(
            file_path=args.dataset, 
            client_id=args.client_id
        )
        
        logger.info(f"✅ Dataset loaded successfully:")
        logger.info(f"   - Training samples: {len(X_train)}")
        logger.info(f"   - Test samples: {len(X_test)}")
        logger.info(f"   - Number of features: {X_train.shape[1]}")
        logger.info(f"   - Global scaler: {'Loaded' if global_scaler else 'Missing'}")
        
        # Initialize the improved model
        input_size = X_train.shape[1]
        model = ImprovedDiabetesModel(input_size=input_size)
        
        logger.info("\n🧠 Fixed Model Architecture:")
        logger.info(model)
        
        # Initialize Flower client
        client = FixedFlowerClient(model, X_train, y_train, X_test, y_test, args.client_id, global_scaler)
        
        # Print connection information
        logger.info("\n🔧 Client Configuration:")
        logger.info("-" * 40)
        logger.info(f"   Client ID: {args.client_id}")
        logger.info(f"   Server: {args.server_address}")
        logger.info(f"   Batch Size: {args.batch_size}")
        logger.info(f"   Device: {client.device}")
        logger.info(f"   Global Scaler: {'✅ Loaded' if global_scaler else '❌ Missing'}")
        logger.info(f"   Features: Data Augmentation, Strong Regularization, Simplified Architecture")
        logger.info("-" * 40)
        
        # Start Flower client
        logger.info("\n🌐 Connecting to server...")
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
        
        logger.info("🎯 Client completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in client: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    run_client()
