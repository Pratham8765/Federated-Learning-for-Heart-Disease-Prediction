import flwr as fl
import torch.nn as nn
import torch
import socket
import time
from zeroconf import ServiceInfo, Zeroconf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define the Global Model Architecture
# This should match the client's model!
class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.layer(x)

# Heart Disease dataset has 13 features and 2 classes
INPUT_SIZE = 13  # Updated to match the Heart Disease dataset
NUM_CLASSES = 2
initial_model = SimpleModel(INPUT_SIZE, NUM_CLASSES)

# Function to get initial weights
def get_initial_parameters(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]

# 2. Define the Strategy
# FedAvg is the standard aggregation algorithm
strategy = fl.server.strategy.FedAvg(
    # Min number of clients to be sampled for training
    min_fit_clients=1, 
    # Min number of clients required to start a round
    min_available_clients=1, 
    # The initial global model weights
    initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters(initial_model)),
)

def register_service(port):
    """Register the Flower server using mDNS/Zeroconf"""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        service_info = ServiceInfo(
            "_http._tcp.local.",
            "Flower Server._http._tcp.local.",
            addresses=[socket.inet_aton(ip) for ip in socket.gethbyname_ex(socket.gethostname())[2]],
            port=port,
            properties={'path': '/'},
            server=f"{hostname}.local.",
        )
        
        zeroconf = Zeroconf()
        zeroconf.register_service(service_info)
        logger.info(f"Registered mDNS service at {local_ip}:{port}")
        return zeroconf
    except Exception as e:
        logger.warning(f"Could not register mDNS service: {e}")
        return None

# Start the Flower Server
def main():
    port = 8080
    
    # Register mDNS service
    zeroconf = register_service(port)
    
    try:
        print("Starting Flower server...")
        print("Clients can connect using one of these methods:")
        print(f"1. Direct IP: Connect to port {port}")
        print("2. mDNS: Look for 'Flower Server' on your local network")
        
        # Start the server on all interfaces (0.0.0.0)
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=5),
            strategy=strategy,
            server=fl.server.Server(client_manager=fl.server.SimpleClientManager(), strategy=strategy)
        )
    finally:
        # Clean up mDNS service
        if zeroconf:
            zeroconf.close()

if __name__ == "__main__":
    main()
