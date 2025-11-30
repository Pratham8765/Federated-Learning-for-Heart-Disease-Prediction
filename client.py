# client.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import flwr as fl
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_heart_disease_data(test_size=0.2):
    """Load and preprocess the Heart Disease dataset."""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        df = pd.read_csv(url, names=column_names, na_values='?')
    except Exception as e:
        logger.error(f"Could not load the dataset: {e}")
        raise

    # Handle missing values and convert target to binary
    df = df.dropna()
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    # Prepare features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test

class HeartDiseaseModel(nn.Module):
    def __init__(self, input_size=13, num_classes=2):
        super(HeartDiseaseModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layer(x)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Training configuration
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        
        # Train the model
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(self.X_train), batch_size):
                batch_x = self.X_train[i:i+batch_size].to(self.device)
                batch_y = self.y_train[i:i+batch_size].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        
        return self.get_parameters(), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            X_test, y_test = self.X_test.to(self.device), self.y_test.to(self.device)
            outputs = self.model(X_test)
            loss = self.criterion(outputs, y_test)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / len(y_test)
        
        return float(loss), len(self.X_test), {"accuracy": accuracy}

def main():
    # Load data
    try:
        X_train, X_test, y_train, y_test = load_heart_disease_data()
        logger.info(f"Loaded Heart Disease dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Initialize model
    model = HeartDiseaseModel(input_size=X_train.shape[1])
    
    # Start client
    client = FlowerClient(model, X_train, y_train, X_test, y_test)
    
    # Connect to server
    logger.info("Connecting to server at localhost:8080...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )

if __name__ == "__main__":
    main()