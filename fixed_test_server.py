# fixed_test_server.py
import requests
import time
import argparse
import sys

def test_prediction(server_ip, server_port, test_data, expected_risk):
    """Test prediction with corrected logic matching server thresholds"""
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
            if 'risk_percentage' in result and 'risk_level' in result:
                risk_pct = float(result['risk_percentage'].rstrip('%'))
                actual_risk_level = result['risk_level']
                
                # Determine expected risk level based on SERVER thresholds
                if risk_pct < 20:
                    expected_label = "Low"
                elif 20 <= risk_pct < 50:
                    expected_label = "Moderate"
                else:
                    expected_label = "High"
                
                # Check if server's risk level matches our calculated expectation
                server_logic_correct = actual_risk_level == expected_label
                
                # Check if it matches the test expectation
                test_passed = actual_risk_level.lower() == expected_risk.lower()
                
                print(f"\n📈 Risk Analysis:")
                print(f"   Risk Percentage: {risk_pct:.2f}%")
                print(f"   Server Risk Level: {actual_risk_level}")
                print(f"   Expected by Threshold: {expected_label}")
                print(f"   Test Expected: {expected_risk}")
                
                print(f"\n{'='*50}")
                if server_logic_correct:
                    print("✅ Server Logic: CORRECT")
                else:
                    print("❌ Server Logic: INCORRECT - Server misclassified")
                
                if test_passed:
                    print("✅ Test Result: PASSED")
                else:
                    print("❌ Test Result: FAILED - Mismatch with test expectation")
                print('='*50)
                
                return test_passed
            else:
                print("\n❌ Invalid response format - missing required fields")
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
    print("\n" + "="*80)
    print(f"🧪 FIXED Diabetes Prediction Tests")
    print(f"   Server: {server_ip}:{server_port}")
    print(f"   Thresholds: Low < 20%, Moderate 20-50%, High > 50%")
    print("="*80)

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

    # Run tests with detailed analysis
    print("\n🔍 Testing High Risk Prediction (Glucose: 183)...")
    high_risk_result = test_prediction(server_ip, server_port, high_risk_data, "High")

    print("\n🔍 Testing Low Risk Prediction (Glucose: 89)...")
    low_risk_result = test_prediction(server_ip, server_port, low_risk_data, "Low")
    
    print("\n🔍 Testing Borderline Risk Prediction (Glucose: 120)...")
    borderline_result = test_prediction(server_ip, server_port, borderline_data, "Moderate")

    # Summary
    print("\n" + "="*80)
    print("📊 FINAL TEST RESULTS:")
    print(f"   High Risk Test:     {'✅ PASSED' if high_risk_result else '❌ FAILED'}")
    print(f"   Low Risk Test:      {'✅ PASSED' if low_risk_result else '❌ FAILED'}")
    print(f"   Borderline Test:    {'✅ PASSED' if borderline_result else '❌ FAILED'}")
    
    overall_success = high_risk_result and low_risk_result and borderline_result
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if not overall_success:
        print("\n💡 Recommendations:")
        if not high_risk_result:
            print("   - Model may be under-confident (High risk case not classified as High)")
            print("   - Consider reducing High threshold to >40% temporarily")
        if not low_risk_result:
            print("   - Model may be over-predicting risk")
        if not borderline_result:
            print("   - Model calibration needs adjustment")
    
    print("="*80 + "\n")
    
    return overall_success

def main():
    parser = argparse.ArgumentParser(description='Fixed Diabetes Prediction Server Test')
    parser.add_argument('--server-ip', type=str, default='127.0.0.1', 
                      help='Server IP address (default: 127.0.0.1)')
    parser.add_argument('--server-port', type=int, default=5000, 
                      help='Server port (default: 5000)')
    
    args = parser.parse_args()
    
    print("🚀 Starting FIXED Diabetes Prediction Tests")
    print(f"🔗 Server: {args.server_ip}:{args.server_port}")
    print("⏳ Testing server connection...")
    
    # Test server connection
    try:
        response = requests.get(f"http://{args.server_ip}:{args.server_port}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server connection established")
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure the server is running and accessible")
        sys.exit(1)
    
    # Run the tests
    success = run_tests(args.server_ip, args.server_port)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
