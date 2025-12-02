import time
import flwr as fl
import torch.nn as nn
import torch
import socket
import logging
from typing import Dict, List, Optional, Tuple
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.common import Parameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the Global Model Architecture
class SimpleModel(nn.Module):
    def __init__(self, input_size=13, num_classes=2):
        super(SimpleModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.layer(x)

# Initialize model
model = SimpleModel()

# Helper functions
def get_initial_parameters(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

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

# Define strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=1,
    min_available_clients=1,
    min_evaluate_clients=1,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(model)),
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
            # This is a simplified example - in a real application, you would:
            # 1. Preprocess the input data
            # 2. Use your trained model to get a prediction
            # 3. Convert the model's output to a probability/percentage
            
            # For demonstration, we'll calculate a simple risk score based on common factors
            risk_score = 0
            
            # Age factor (increases risk after 45)
            age = data.get('age', 0)
            if age >= 45:
                risk_score += 15
            
            # Sex (male is higher risk)
            if data.get('sex') == 1:  # 1 typically represents male
                risk_score += 5
                
            # Cholesterol (higher than 200 is concerning)
            chol = data.get('chol', 0)
            if chol > 200:
                risk_score += 15
            
            # Blood pressure (higher than 120/80 is concerning)
            trestbps = data.get('trestbps', 0)
            if trestbps > 130:
                risk_score += 15
            
            # Other risk factors
            if data.get('cp', 0) > 0:  # Chest pain
                risk_score += 15
            if data.get('fbs', 0) > 120:  # Fasting blood sugar
                risk_score += 10
            if data.get('exang', 0) == 1:  # Exercise induced angina
                risk_score += 10
            if data.get('oldpeak', 0) > 0:  # ST depression
                risk_score += 10
            if data.get('slope', 0) <= 1:  # Slope of peak exercise ST segment
                risk_score += 5
                
            # Cap the risk score at 100%
            risk_percentage = min(100, risk_score)
            
            # Determine risk level
            if risk_percentage < 30:
                risk_level = "Low"
            elif risk_percentage < 70:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            return {
                "status": "success",
                "risk_percentage": f"{risk_percentage}%",
                "risk_level": risk_level,
                "interpretation": f"Based on the provided health data, the patient has a {risk_percentage}% risk of heart disease. This is considered {risk_level} risk.",
                "recommendation": "Please consult with a healthcare professional for a comprehensive evaluation." if risk_percentage > 30 else "Maintain a healthy lifestyle and regular check-ups."
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Start FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=5001)
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    print("Starting Flower server...")
    print("FastAPI server running on http://10.26.65.217:5001")
    print("Server is running. Press Ctrl+C to stop.")
    
    try:
        # Start Flower server
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(
                num_rounds=5,
                round_timeout=None,
            ),
            strategy=strategy
        )
        
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