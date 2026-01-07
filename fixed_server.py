# fixed_server.py
import time
import flwr as fl
import torch.nn as nn
import torch
import socket
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedProx
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.common import Parameters, Scalar, FitRes, Parameters, NDArrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified Model Architecture with Reduced Regularization
class TunedDiabetesModel(nn.Module):
    def __init__(self, input_size=8, num_classes=2):
        super(TunedDiabetesModel, self).__init__()
        # Simplified architecture with reduced regularization
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 16),  # Reduced from 32
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)  # REDUCED from 0.6
        )
        self.layer2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2)  # REDUCED from 0.5
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

# Load global scaler
def load_global_scaler():
    """Load the global scaler for consistent feature scaling"""
    try:
        scaler = joblib.load('global_scaler.pkl')
        logger.info("✅ Global scaler loaded successfully")
        return scaler
    except FileNotFoundError:
        logger.warning("⚠️ Global scaler not found, creating new one...")
        # Create scaler on the fly if not exists
        df = pd.read_csv('diabetes_non_negative_part1_2000.csv')
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        X = df[feature_columns].values if all(col in df.columns for col in feature_columns) else df.iloc[:, :8].values
        
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, 'global_scaler.pkl')
        logger.info("🔧 Created and saved new global scaler")
        return scaler
    except Exception as e:
        logger.error(f"Error loading global scaler: {e}")
        return None

# Load server-side validation data
def load_server_validation_data():
    """Load a separate validation dataset for server-side evaluation"""
    try:
        # Use part2 as validation set to ensure data diversity
        df = pd.read_csv('diabetes_non_negative_part2_2000.csv')
        
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        if all(col in df.columns for col in feature_columns):
            X = df[feature_columns].values
            y = df['Outcome'].values if 'Outcome' in df.columns else df.iloc[:, -1].values
        else:
            X = df.iloc[:, :8].values
            y = df.iloc[:, -1].values
            
        # Convert target to binary if needed
        if len(np.unique(y)) > 2:
            y = (y > 0).astype(int)
            
        # Split into train/validation for server
        X_server_train, X_val, y_server_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize using the SAME scaler as training
        global_scaler = load_global_scaler()
        if global_scaler is not None:
            X_val = global_scaler.transform(X_val)
        else:
            # Fallback - create local scaler
            scaler = StandardScaler()
            X_val = scaler.fit_transform(X_val)
            
        # Convert to tensors
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        
        logger.info(f"Server validation set: {len(X_val)} samples")
        return X_val, y_val, global_scaler
        
    except Exception as e:
        logger.warning(f"Could not load server validation data: {e}")
        return None, None, None

# Initialize tuned model
model = TunedDiabetesModel(input_size=8, num_classes=2)

# Initialize model weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()

model.apply(init_weights)

# Load global scaler and server validation data
global_scaler = load_global_scaler()
X_val, y_val, _ = load_server_validation_data()

# Log model architecture for debugging
logger.info("Tuned Server model architecture:")
logger.info(model)

# Helper functions
def get_initial_parameters(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    return {
        "round": server_round,
        "epochs": 3,  # INCREASED from 1 to combat under-fitting
        "learning_rate": max(0.001 * (0.95 ** server_round), 0.0001),  # Learning rate decay
        "weight_decay": 1e-5,  # REDUCED from 5e-4 to combat under-fitting
    }

def evaluate_config(server_round: int):
    return {"round": server_round}

# Server-side evaluation function
def server_evaluate(model, X_val, y_val):
    """Evaluate model on server-side validation set"""
    if X_val is None or y_val is None:
        return None
        
    model.eval()
    with torch.no_grad():
        outputs = model(X_val)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate metrics
        y_true = y_val.numpy()
        y_pred = predicted.numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        logger.info(f"Server Validation - Accuracy: {accuracy:.4f}, "
                   f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

# Improved FedProx strategy with server-side validation
class ImprovedFedProx(FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = model.to(self.device)
        self.server_val_results = []
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Aggregate parameters using FedProx
        if not results:
            return None, {}
            
        # Call parent's aggregate_fit
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Update global model with new parameters
        if aggregated_parameters is not None:
            parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
                         for k, v in params_dict}
            self.global_model.load_state_dict(state_dict, strict=True)
            
            # Server-side validation
            val_metrics = server_evaluate(self.global_model, X_val, y_val)
            if val_metrics:
                self.server_val_results.append((server_round, val_metrics))
                metrics_aggregated.update({f"server_{k}": v for k, v in val_metrics.items()})
            
        return aggregated_parameters, metrics_aggregated
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # Configure clients with improved hyperparameters
        config = {
            "round": server_round,
            "epochs": 1,  # Keep at 1 to minimize client drift
            "learning_rate": max(0.001 * (0.95 ** server_round), 0.0001),  # Learning rate decay
            "proximal_mu": 0.1,  # FedProx parameter
            "weight_decay": 5e-4,  # Increased regularization
        }
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        config = {"round": server_round}
        clients = client_manager.sample(
            num_clients=min(self.min_evaluate_clients, client_manager.num_available()),
            min_num_clients=self.min_evaluate_clients,
        )
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

# Define improved strategy
strategy = ImprovedFedProx(
    min_fit_clients=2,
    min_available_clients=2,
    min_evaluate_clients=2,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(model)),
    proximal_mu=0.1,
)

def main():
    # Start FastAPI server in a separate thread
    import threading
    from fastapi import FastAPI, HTTPException
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi import Request
    
    # Create FastAPI app
    app = FastAPI()
    
    # Mount static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Setup templates
    import os
    templates = Jinja2Templates(directory="templates")
    
    # Root endpoint to serve the HTML form
    @app.get("/")
    async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
    
    # Add prediction endpoint
    @app.post("/predict")
    async def predict(data: dict):
        try:
            # Convert input data to tensor and make prediction
            input_features = [
                float(data.get('Pregnancies', 0)),
                float(data.get('Glucose', 0)),
                float(data.get('BloodPressure', 0)),
                float(data.get('SkinThickness', 0)),
                float(data.get('Insulin', 0)),
                float(data.get('BMI', 0)),
                float(data.get('DiabetesPedigreeFunction', 0)),
                float(data.get('Age', 0))
            ]
            
            print(f"🔍 Raw input features: {input_features}")
            
            # Apply CONSISTENT scaling using global scaler
            if global_scaler is not None:
                input_features_scaled = global_scaler.transform([input_features])[0]
                print(f"📏 Scaled features: {np.round(input_features_scaled, 4)}")
            else:
                input_features_scaled = input_features
                print("⚠️ No scaler available, using raw features")
            
            # Convert to tensor and predict
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor([input_features_scaled])
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                risk_percentage = round(probabilities[0][1].item() * 100, 2)
                
                print(f"🧠 Model output: {output}")
                print(f"📊 Probabilities: {probabilities}")
                print(f"🎯 Risk percentage: {risk_percentage}%")
                
            # Determine risk level based on medical diabetes risk standards (TEMPORARY CALIBRATION)
            if risk_percentage < 20:
                risk_level = "Low"
                recommendation = "Maintain a healthy lifestyle with regular exercise and balanced diet."
            elif risk_percentage < 40:  # TEMPORARILY LOWERED from 50% to capture under-confident model
                risk_level = "Moderate"
                recommendation = "Consider lifestyle changes and regular monitoring. Consult a healthcare provider."
            else:
                risk_level = "High"
                recommendation = "Please consult with an endocrinologist for a comprehensive evaluation and management plan."
            
            return {
                "status": "success",
                "risk_percentage": f"{risk_percentage}%",
                "risk_level": risk_level,
                "interpretation": f"Based on provided health data, patient has a {risk_percentage}% risk of developing diabetes. This is considered {risk_level} risk.",
                "recommendation": recommendation
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Start FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=5000)
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Use localhost for local testing
    server_ip = '127.0.0.1'
    
    print("\n" + "="*60)
    print(f"🔧 FIXED Diabetes Prediction Server")
    print("="*60)
    print(f"🌐 Flower server running on: {server_ip}:8080")
    print(f"🌐 FastAPI server running on: http://{server_ip}:5000")
    print(f"✅ Features: Simplified architecture, strong regularization, server-side validation")
    print(f"📏 Global Scaler: {'Loaded' if global_scaler else 'Created'}")
    print("="*60)
    
    try:
        # Start Flower server with improved configuration
        history = fl.server.start_server(
            server_address=f"{server_ip}:8080",
            config=fl.server.ServerConfig(
                num_rounds=25,  # Reduced rounds to prevent overfitting
                round_timeout=120,
            ),
            strategy=strategy
        )
        
        # Save final model
        torch.save(model.state_dict(), 'fixed_global_model.pth')
        print("\n🎯 Fixed global model saved to fixed_global_model.pth")
        
        # Print server validation results
        if strategy.server_val_results:
            print("\n" + "="*60)
            print("📊 SERVER-SIDE VALIDATION RESULTS:")
            print("="*60)
            for round_num, metrics in strategy.server_val_results:
                print(f"Round {round_num}: Accuracy={metrics['accuracy']:.4f}, "
                      f"F1={metrics['f1_score']:.4f}")
        
        print("\n🚀 Training completed! Starting server test...")
        print("="*60)
        
        # Import and run test
        import subprocess
        import sys
        python = sys.executable
        subprocess.Popen([python, "test_server.py"])
        
        print("\n🌐 Server is running and ready for predictions.")
        print("Press Ctrl+C to stop the server.")
        print("="*60 + "\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")

if __name__ == "__main__":
    main()
