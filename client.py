# client.py
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
from fastapi import FastAPI, HTTPException
import uvicorn
import threading
import time
import sys
import argparse
import socket
import torch.utils.data as data

# Server IP Configuration - UPDATE THIS FOR NETWORK DEPLOYMENT
SERVER_IP = "10.133.98.49"  # Server's actual IP address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("client.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://10.26.65.217:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_diabetes_data(file_path='diabetes_non_negative_part1_2000.csv', test_size=0.2):
    """Load and preprocess the Diabetes dataset."""
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Check if the file was loaded correctly
        if df.empty:
            raise ValueError("The dataset is empty. Please check the file path.")
            
        # Display basic info about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info("\nDataset info:")
        logger.info(df.info())
        logger.info("\nFirst few rows of the dataset:")
        logger.info(df.head())
        
        # Define expected features based on the database columns (excluding 'Outcome')
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Check if all expected features are present
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            logger.warning(f"Warning: Missing expected features: {missing_features}")
            logger.warning("Using first 8 columns as features")
            X = df.iloc[:, :8].values  # Use first 8 columns as features
        else:
            X = df[feature_columns].values  # Use specified features if available
            
        # Target is the 'Outcome' column if it exists, otherwise use the last column
        if 'Outcome' in df.columns:
            y = df['Outcome'].values
        else:
            y = df.iloc[:, -1].values
        
        logger.info(f"Loaded {X.shape[1]} features and {y.shape[0]} samples")
        
        # Convert target to binary if it's not already
        if len(np.unique(y)) > 2:
            # If more than 2 classes, convert to binary (1 if positive, 0 otherwise)
            y = (y > 0).astype(int)
            
    except Exception as e:
        logger.error(f"Error loading the dataset: {e}")
        raise
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test

class DiabetesModel(nn.Module):
    def __init__(self, input_size=8, num_classes=2):
        super(DiabetesModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.3)
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
        x = self.layer3(x)
        return self.output(x)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss function with label smoothing
        self.label_smoothing = 0.1  # 10% label smoothing
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.0017,  # Updated learning rate
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,  # Reduce LR by half when validation loss stops improving
            patience=2  # Number of epochs with no improvement
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 3
        self.no_improve_epochs = 0
        
        logger.info(f"Client initialized on device: {self.device}")
    
    def get_parameters(self, config=None):
        # Return model parameters as a list of numpy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        try:
            # Get current model state dict
            state_dict = self.model.state_dict()
            
            # Create a mapping of parameter names to their values
            param_dict = {name: param for name, param in zip(state_dict.keys(), parameters)}
            
            # Update the state dict with matching parameters
            for name, param in state_dict.items():
                if name in param_dict:
                    # Get the parameter from the server
                    server_param = param_dict[name]
                    
                    # Convert to tensor if it's a numpy array
                    if isinstance(server_param, np.ndarray):
                        server_param = torch.from_numpy(server_param)
                    
                    # If shapes match, use as is
                    if param.shape == server_param.shape:
                        state_dict[name] = server_param.to(self.device)
                    # Handle common shape mismatches
                    elif len(server_param.shape) > len(param.shape) and server_param.shape[0] == 1:
                        # Case: Server sent [1, N] but we need [N]
                        state_dict[name] = server_param.squeeze(0).to(self.device)
                    elif len(server_param.shape) > len(param.shape) and server_param.shape[1] == 1:
                        # Case: Server sent [N, 1] but we need [N]
                        state_dict[name] = server_param.squeeze(1).to(self.device)
                    elif server_param.numel() == param.numel():
                        # Case: Same number of elements but different shapes
                        state_dict[name] = server_param.reshape_as(param).to(self.device)
                    else:
                        # For other cases, log a warning but try to make it work
                        logger.warning(f"Shape mismatch for {name}: expected {param.shape}, got {server_param.shape}")
                        if server_param.numel() >= param.numel():
                            # If server sent more parameters, take the first ones that fit
                            state_dict[name] = server_param.flatten()[:param.numel()].reshape(param.shape).to(self.device)
                        else:
                            # If server sent fewer parameters, pad with zeros
                            temp = torch.zeros_like(param, device=self.device)
                            flat_temp = temp.flatten()
                            flat_temp[:server_param.numel()] = server_param.flatten()
                            state_dict[name] = flat_temp.reshape(param.shape)
            
            # Load the updated state dict
            self.model.load_state_dict(state_dict, strict=False)
            
        except Exception as e:
            logger.error(f"Error in set_parameters: {str(e)}")
            logger.error("Attempting to load parameters with simplified method...")
            # As a last resort, try to load as many parameters as possible
            try:
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if i < len(parameters):
                        param.data = torch.from_numpy(parameters[i]).to(self.device)
                logger.info("Successfully loaded parameters with simplified method")
            except Exception as e2:
                logger.critical(f"Critical error in parameter loading: {str(e2)}")
                # If all else fails, reinitialize the model
                logger.info("Reinitializing model weights...")
                self.model.apply(self._init_weights)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Training configuration with FedProx support
        epochs = config.get("epochs", 1)  # Keep at 1 to prevent client drift
        batch_size = config.get("batch_size", 32)
        proximal_mu = config.get("proximal_mu", 0.2)  # Increased from 0.1
        weight_decay = config.get("weight_decay", 1e-4)
        
        # Save global parameters for FedProx
        if proximal_mu > 0:
            global_params = [p.detach().clone() for p in self.model.parameters()]
        
        # Create data loader for batching
        train_dataset = data.TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train)
        )
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True  # Drop last batch if smaller than batch_size
        )
        
        # Training loop
        self.model.train()
        best_params = None
        best_val_loss = float('inf')
        patience = 3
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            # Training phase
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # Standard cross-entropy loss
                loss = self.criterion(outputs, batch_y)
                
                # L2 regularization (weight decay is already in the optimizer, but we add it explicitly for clarity)
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += weight_decay * l2_reg
                
                # Label smoothing
                if self.label_smoothing > 0:
                    smooth_loss = -F.log_softmax(outputs, dim=1).mean(dim=1).mean()
                    loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss
                
                # Add FedProx term if enabled
                if proximal_mu > 0:
                    proximal_term = 0.
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += (local_param - global_param).norm(2)
                    loss += (proximal_mu / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            
            # Validation phase
            val_loss, val_metrics = self._evaluate_validation()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Log training progress
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {epoch_loss/len(self.X_train):.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = [p.detach().cpu().numpy() for p in self.model.parameters()]
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}, loading best model...")
                    if best_params is not None:
                        self.set_parameters(best_params)
                    break
        
        return self.get_parameters(), len(self.X_train), {}
    
    def _evaluate_validation(self):
        """Helper method to evaluate on validation set"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_test_tensor = torch.LongTensor(self.y_test).to(self.device)
            
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
            X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
            y_test_tensor = torch.LongTensor(self.y_test).to(self.device)
            
            outputs = self.model(X_test_tensor)
            loss = self.criterion(outputs, y_test_tensor).item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test_tensor)
            
            # Calculate precision, recall, f1
            y_true = y_test_tensor.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            # Handle case when all predictions are the same class
            if len(np.unique(y_pred)) == 1:
                precision = recall = f1 = 0.0
            else:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Log evaluation metrics
            logger.info(f"Evaluation - Loss: {loss:.4f}, "
                       f"Accuracy: {accuracy:.4f}, "
                       f"Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, "
                       f"F1: {f1:.4f}")
        
        return float(loss), len(self.X_test), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

def get_local_ip():
    """Get local IP address for the machine."""
    try:
        # Connect to an external server to get the local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logger.warning(f"Could not determine local IP address: {e}")
        return "0.0.0.0"

def run_client():
    # Parse command line arguments for the main process
    parser = argparse.ArgumentParser(description='Federated Learning Client for Diabetes Prediction')
    parser.add_argument('--server-address', type=str, default=f"{SERVER_IP}:8080",
                      help='Address of the server in the format host:port (default: SERVER_IP:8080)')
    parser.add_argument('--client-id', type=int, required=True,
                      help='Unique identifier for this client (required)')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    
    args = parser.parse_args()
    
    # Dynamic data loading based on client ID
    if args.client_id == 1:
        dataset_file = 'diabetes_non_negative_part1_2000.csv'
        logger.info(f"Client {args.client_id}: Loading dataset part 1")
    elif args.client_id == 2:
        dataset_file = 'diabetes_non_negative_part2_2000.csv'
        logger.info(f"Client {args.client_id}: Loading dataset part 2")
    else:
        # Default to part1 for any other client ID
        dataset_file = 'diabetes_non_negative_part1_2000.csv'
        logger.info(f"Client {args.client_id}: Loading default dataset part 1")
    
    try:
        logger.info("=" * 50)
        logger.info(f"Starting Federated Learning Client (ID: {args.client_id})")
        logger.info("=" * 50)
        
        # Load and preprocess the dataset
        X_train, X_test, y_train, y_test = load_diabetes_data(file_path=dataset_file)
        
        # Log dataset information
        logger.info(f"Dataset loaded successfully:")
        logger.info(f"- Training samples: {len(X_train)}")
        logger.info(f"- Test samples: {len(X_test)}")
        logger.info(f"- Number of features: {X_train.shape[1]}")
        
        # Initialize the model with the correct input size
        input_size = X_train.shape[1]
        model = DiabetesModel(input_size=input_size)
        
        # Log model architecture
        logger.info("\nModel Architecture:")
        logger.info(model)
        
        # Initialize Flower client
        client = FlowerClient(model, X_train, y_train, X_test, y_test)
        
        # Print connection information
        logger.info("\nClient Configuration:")
        logger.info("-" * 20)
        logger.info(f"Client ID: {args.client_id}")
        logger.info(f"Server: {args.server_address}")
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Device: {client.device}")
        logger.info("-" * 20)
        
        # Start Flower client
        logger.info("\nConnecting to server...")
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
        
        logger.info("Client finished successfully!")
        
    except Exception as e:
        logger.error(f"Error in client: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    run_client()