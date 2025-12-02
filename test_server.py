import requests
import time
import socket

def is_port_open(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((ip, port))
        s.close()
        return True
    except:
        return False

def test_prediction():
    test_data = {
        "age": 57,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "chol": 236,
        "fbs": 0,
        "restecg": 0,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 1,
        "thal": 1
    }
    
    server_ip = "10.26.65.217"
    server_port = 5001
    
    print("\n" + "="*70)
    print("🔍 Testing server connection...")
    
    # Check if port is open
    if not is_port_open(server_ip, server_port):
        print(f"❌ Cannot connect to {server_ip}:{server_port}. Is the server running?")
        return False
        
    print(f"✅ Port {server_port} is open on {server_ip}")
    print("\nSending test data:", test_data)
    
    try:
        url = f"http://{server_ip}:{server_port}/predict"
        print(f"\nSending request to: {url}")
        
        response = requests.post(url, json=test_data, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        result = response.json()
        print("\n✅ Server response:", result)
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status code: {e.response.status_code}")
            print("Response content:", e.response.text)
        return False

if __name__ == "__main__":
    print("🚀 Starting server test...")
    print("This will test the prediction endpoint after a short delay.")
    
    # Wait for server to start
    wait_time = 10
    print(f"⏳ Waiting {wait_time} seconds for server to initialize...")
    time.sleep(wait_time)
    
    if test_prediction():
        print("\n✨ Server test successful!")
    else:
        print("\n❌ Server test failed!")
    
    print("="*70 + "\n")
