import time
import flwr as fl
import torch.nn as nn
import torch
import socket
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedProx
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.common import Parameters, Scalar, FitRes, Parameters, NDArrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the Global Model Architecture
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

# Initialize model with 8 features to match the client's dataset
model = DiabetesModel(input_size=8, num_classes=2)  # 8 features, 2 classes (diabetes/no diabetes)

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

# Log model architecture for debugging
logger.info("Server model architecture:")
logger.info(model)

# Helper functions
def get_initial_parameters(model):
    # Make sure to detach and clone to avoid any potential in-place modifications
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    return {
        "round": server_round,
        "epochs": 1,
    }

def evaluate_config(server_round: int):
    return {"round": server_round}

# Define FedProx strategy for non-IID data
class CustomFedProx(FedProx):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_test_loader = None  # Will be set before evaluation
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.global_model = model.to(self.device)
    
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
            # Convert Parameters to list of numpy arrays
            parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Create state dict with model's keys and new parameters
            params_dict = zip(self.global_model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v 
                         for k, v in params_dict}
            self.global_model.load_state_dict(state_dict, strict=True)
            
        return aggregated_parameters, metrics_aggregated
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # Configure clients with proximal term mu
        config = {
            "round": server_round,
            "epochs": 1,  # Reduced to 1 to minimize client drift
            "proximal_mu": 0.1,  # FedProx parameter
        }
        return super().configure_fit(server_round, parameters, client_manager)
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        # Only evaluate on a subset of clients
        config = {"round": server_round}
        clients = client_manager.sample(
            num_clients=min(self.min_evaluate_clients, client_manager.num_available()),
            min_num_clients=self.min_evaluate_clients,
        )
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

# Define strategy with FedProx - Updated to require 2 clients
strategy = CustomFedProx(
    min_fit_clients=2,        # Require 2 clients for training
    min_available_clients=2,  # Need 2 clients to be available
    min_evaluate_clients=2,   # Evaluate on both clients
    fraction_fit=1.0,         # Use 100% of available clients for training
    fraction_evaluate=1.0,    # Evaluate on 100% of available clients
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(model)),
    proximal_mu=0.1,  # FedProx parameter
)

class PersistentServer(fl.server.Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_running = True

    def run(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = super().run(num_rounds, timeout)
        print("\nTraining completed. Keeping server alive for new connections...")
        while self.keep_running:
            time.sleep(1)
        return history

# ... (previous imports and model definition remain the same) ...

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
            # Using only 9 features as expected by the model
            input_features = [
                float(data.get('Pregnancies', 0)),
                float(data.get('Glucose', 0)),
                float(data.get('BloodPressure', 0)),
                float(data.get('SkinThickness', 0)),
                float(data.get('Insulin', 0)),
                float(data.get('BMI', 0)),
                float(data.get('DiabetesPedigreeFunction', 0)),
                float(data.get('Age', 0))
                # Removed 'Outcome' as it's the target variable
            ]
            
            # Convert to tensor and predict
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor([input_features])
                output = model(input_tensor)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(output, dim=1)
                # Get the probability of the positive class (class 1)
                risk_percentage = round(probabilities[0][1].item() * 100, 2)
                
                # Debug information
                print(f"Model output: {output}")
                print(f"Probabilities: {probabilities}")
                print(f"Risk percentage: {risk_percentage}%")
                
            # Determine risk level based on probability with better thresholds
            if risk_percentage < 25:
                risk_level = "Low"
                recommendation = "Maintain a healthy lifestyle with regular exercise and balanced diet."
            elif risk_percentage < 75:
                risk_level = "Moderate"
                recommendation = "Consider lifestyle changes and regular monitoring. Consult a healthcare provider."
            else:
                risk_level = "High"
                recommendation = "Please consult with an endocrinologist for a comprehensive evaluation and management plan."
            
            return {
                "status": "success",
                "risk_percentage": f"{risk_percentage}%",
                "risk_level": risk_level,
                "interpretation": f"Based on the provided health data, the patient has a {risk_percentage}% risk of developing diabetes. This is considered {risk_level} risk.",
                "recommendation": recommendation
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Start FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=5000)  # Server API on port 5000
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Bind to 0.0.0.0 for network deployment
    server_ip = '0.0.0.0'
    
    # Get and display the server's machine IP address
    import socket
    machine_ip = socket.gethostbyname(socket.gethostname())
    
    print("\n" + "="*50)
    print(f"Starting Diabetes Prediction Server")
    print("="*50)
    print(f"Flower server binding to: {server_ip}:8080")
    print(f"Machine IP Address: {machine_ip}")
    print(f"FastAPI server running on: http://{server_ip}:5000")
    print(f"\nClients should connect to: {machine_ip}:8080")
    print("\nServer is running. Will work with 1 or 2 clients...")
    print("="*50)
    
    try:
        # Start Flower server with more communication rounds
        history = fl.server.start_server(
            server_address=f"{server_ip}:8080",
            config=fl.server.ServerConfig(
                num_rounds=30,  # Increased from 10 to 30 for better convergence
                round_timeout=120,  # 2 minute timeout per round
            ),
            strategy=strategy
        )
        
        # Save final model
        torch.save(model.state_dict(), 'global_model_final.pth')
        print("\nFinal global model saved to global_model_final.pth")
        
        # After training completes
        print("\n" + "="*50)
        print("Training completed! Starting server test...")
        print("="*50)
        
        # Import and run test
        import subprocess
        import sys
        
        # Use the same Python interpreter that's running the server
        python = sys.executable
        subprocess.Popen([python, "test_server.py"])
        
        # Keep server running for predictions
        print("\nServer is running and ready for predictions.")
        print("Press Ctrl+C to stop the server.")
        print("Test the prediction endpoint with: python test_server.py")
        print("="*50 + "\n")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    main()