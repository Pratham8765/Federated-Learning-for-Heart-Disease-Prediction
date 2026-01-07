# test_server.py
import requests
import time
import argparse
import sys

def test_prediction(server_ip, server_port, test_data, expected_risk):
    url = f"http://{server_ip}:{server_port}/predict"
    try:
        print(f"\n🌐 Sending request to: {url}")
        print(f"   Data: {test_data}")
        response = requests.post(url, json=test_data, timeout=10)
        
        print(f"\n📥 Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("📊 Response data:", result)
            
            # Check if the prediction matches the expected risk
            if 'risk_percentage' in result:
                risk_pct = float(result['risk_percentage'].rstrip('%'))
                predicted_risk = "High" if risk_pct >= 50 else "Low"
                passed = predicted_risk.lower() == expected_risk.lower()
                print(f"\n✅ Test {'PASSED' if passed else 'FAILED'}")
                print(f"   Expected: {expected_risk} risk")
                print(f"   Got: {predicted_risk} risk ({risk_pct:.2f}%)")
                print(f"   Details: {result.get('interpretation', 'No interpretation')}")
                return passed
            else:
                print("\n❌ Invalid response format - missing 'risk_percentage'")
                print(f"   Response: {result}")
                return False
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Connection error: {str(e)}")
        print(f"   URL: {url}")
        print("   Make sure the server is running and accessible")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_tests(server_ip, server_port):
    print("\n" + "="*70)
    print(f"🧪 Starting Diabetes Prediction Tests")
    print(f"   Server: {server_ip}:{server_port}")
    print("="*70)

    # High Risk Test Case (values indicating potential diabetes)
    high_risk_data = {
        'Pregnancies': 8,
        'Glucose': 183,
        'BloodPressure': 64,
        'SkinThickness': 0,
        'Insulin': 0,
        'BMI': 35.2,
        'DiabetesPedigreeFunction': 0.672,
        'Age': 50
    }

    # Low Risk Test Case (normal/healthy values)
    low_risk_data = {
        'Pregnancies': 1,
        'Glucose': 89,
        'BloodPressure': 66,
        'SkinThickness': 23,
        'Insulin': 94,
        'BMI': 22.5,
        'DiabetesPedigreeFunction': 0.167,
        'Age': 25
    }
    
    # Borderline Test Case (moderate risk values)
    borderline_data = {
        'Pregnancies': 3,
        'Glucose': 120,
        'BloodPressure': 75,
        'SkinThickness': 30,
        'Insulin': 150,
        'BMI': 28.0,
        'DiabetesPedigreeFunction': 0.4,
        'Age': 35
    }

    # Run tests
    print("\n🔍 Testing High Risk Prediction...")
    high_risk_result = test_prediction(server_ip, server_port, high_risk_data, "High")

    print("\n🔍 Testing Low Risk Prediction...")
    low_risk_result = test_prediction(server_ip, server_port, low_risk_data, "Low")
    
    print("\n🔍 Testing Borderline Risk Prediction...")
    borderline_result = test_prediction(server_ip, server_port, borderline_data, "Moderate")

    # Summary
    print("\n" + "="*70)
    print("📊 Test Results:")
    print(f"   High Risk Test:     {'✅ PASSED' if high_risk_result else '❌ FAILED'}")
    print(f"   Low Risk Test:      {'✅ PASSED' if low_risk_result else '❌ FAILED'}")
    print(f"   Borderline Test:    {'✅ PASSED' if borderline_result else '❌ FAILED'}")
    print("="*70 + "\n")

    return high_risk_result and low_risk_result and borderline_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Diabetes Prediction Server')
    parser.add_argument('--server-ip', type=str, default="127.0.0.1",
                      help='Server IP address (default: 127.0.0.1)')
    parser.add_argument('--server-port', type=int, default=5000,
                      help='Server port (default: 5000)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Diabetes Prediction Tests")
    print(f"🔗 Server: {args.server_ip}:{args.server_port}")
    print("⏳ Testing server connection...")
    
    # Wait a moment for the server to be ready
    time.sleep(2)
    
    success = run_tests(args.server_ip, args.server_port)
    
    if success:
        print("✨ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)