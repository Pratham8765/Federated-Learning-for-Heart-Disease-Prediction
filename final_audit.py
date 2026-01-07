# final_audit.py
import requests
import json
import time
import numpy as np
from typing import Dict, List, Tuple
import sys

class FinalAudit:
    """Final Acceptance Test for Trained Federated Learning Model"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:5000"):
        self.server_url = server_url
        self.test_results = {}
        
    def create_synthetic_patients(self) -> Dict[str, Dict]:
        """Create 3 synthetic patients for common sense testing"""
        patients = {
            "Patient_A_Healthy": {
                "name": "Patient A (Healthy)",
                "data": {
                    'Pregnancies': 1,
                    'Glucose': 80,      # Low glucose
                    'BloodPressure': 70,
                    'SkinThickness': 20,
                    'Insulin': 80,
                    'BMI': 20.0,        # Low BMI
                    'DiabetesPedigreeFunction': 0.3,
                    'Age': 25           # Young
                },
                "expected_risk": "Low",
                "expected_max_risk": 20.0  # Should be < 20%
            },
            "Patient_B_Borderline": {
                "name": "Patient B (Borderline)",
                "data": {
                    'Pregnancies': 2,
                    'Glucose': 140,     # Moderate glucose
                    'BloodPressure': 80,
                    'SkinThickness': 30,
                    'Insulin': 140,
                    'BMI': 28.0,       # Moderate BMI
                    'DiabetesPedigreeFunction': 0.5,
                    'Age': 45           # Middle-aged
                },
                "expected_risk": "Moderate",
                "expected_range": (20.0, 50.0)  # Should be 20-50%
            },
            "Patient_C_Severe": {
                "name": "Patient C (Severe)",
                "data": {
                    'Pregnancies': 5,
                    'Glucose': 200,     # High glucose
                    'BloodPressure': 90,
                    'SkinThickness': 40,
                    'Insulin': 200,
                    'BMI': 35.0,       # High BMI
                    'DiabetesPedigreeFunction': 0.8,
                    'Age': 60           # Older
                },
                "expected_risk": "High",
                "expected_min_risk": 70.0  # Should be > 70%
            }
        }
        return patients
    
    def predict_risk(self, patient_data: Dict) -> Tuple[float, str, Dict]:
        """Make prediction and return risk percentage and level"""
        try:
            response = requests.post(
                f"{self.server_url}/predict",
                json=patient_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                risk_pct = float(result['risk_percentage'].rstrip('%'))
                risk_level = result['risk_level']
                return risk_pct, risk_level, result
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def test_common_sense_ranking(self) -> bool:
        """Test 1: Common Sense Ranking - Risk(A) < Risk(B) < Risk(C)"""
        print("\n" + "="*80)
        print("🧪 TEST 1: COMMON SENSE RANKING")
        print("="*80)
        print("Testing: Risk(Healthy) < Risk(Borderline) < Risk(Severe)")
        print()
        
        patients = self.create_synthetic_patients()
        risks = {}
        
        # Get predictions for all patients
        for patient_id, patient_info in patients.items():
            try:
                risk_pct, risk_level, full_result = self.predict_risk(patient_info["data"])
                risks[patient_id] = {
                    "risk_pct": risk_pct,
                    "risk_level": risk_level,
                    "full_result": full_result
                }
                
                print(f"📊 {patient_info['name']}:")
                print(f"   Risk Percentage: {risk_pct:.2f}%")
                print(f"   Risk Level: {risk_level}")
                print(f"   Expected: {patient_info['expected_risk']}")
                print()
                
            except Exception as e:
                print(f"❌ ERROR predicting {patient_info['name']}: {e}")
                return False
        
        # Check ranking: Healthy < Borderline < Severe
        healthy_risk = risks["Patient_A_Healthy"]["risk_pct"]
        borderline_risk = risks["Patient_B_Borderline"]["risk_pct"]
        severe_risk = risks["Patient_C_Severe"]["risk_pct"]
        
        ranking_correct = (healthy_risk < borderline_risk < severe_risk)
        
        print("🔍 RANKING ANALYSIS:")
        print(f"   Healthy (A): {healthy_risk:.2f}%")
        print(f"   Borderline (B): {borderline_risk:.2f}%")
        print(f"   Severe (C): {severe_risk:.2f}%")
        print()
        
        if ranking_correct:
            print("✅ RANKING TEST: PASSED")
            print("   ✅ Risk(A) < Risk(B) < Risk(C) - Common sense verified")
        else:
            print("❌ RANKING TEST: FAILED")
            print("   ❌ Model violates common sense ranking!")
            print("   ❌ This indicates a critical model failure")
        
        self.test_results["common_sense_ranking"] = {
            "passed": ranking_correct,
            "risks": risks
        }
        
        return ranking_correct
    
    def test_confidence_gap(self) -> bool:
        """Test 2: Confidence Gap - Ensure decisive predictions"""
        print("\n" + "="*80)
        print("🧪 TEST 2: CONFIDENCE GAP CHECK")
        print("="*80)
        print("Testing: Healthy < 20%, Severe > 70%")
        print()
        
        patients = self.create_synthetic_patients()
        confidence_passed = True
        
        # Test Patient A (should be < 20%)
        try:
            healthy_risk, _, _ = self.predict_risk(patients["Patient_A_Healthy"]["data"])
            healthy_threshold = patients["Patient_A_Healthy"]["expected_max_risk"]
            
            print(f"📊 Patient A (Healthy) Confidence Test:")
            print(f"   Risk: {healthy_risk:.2f}%")
            print(f"   Threshold: < {healthy_threshold}%")
            
            if healthy_risk < healthy_threshold:
                print("✅ HEALTHY CONFIDENCE: PASSED")
            else:
                print("❌ HEALTHY CONFIDENCE: FAILED")
                print("   ❌ Model not confident enough in healthy case")
                confidence_passed = False
            print()
            
        except Exception as e:
            print(f"❌ ERROR testing healthy confidence: {e}")
            confidence_passed = False
        
        # Test Patient C (should be > 70%)
        try:
            severe_risk, _, _ = self.predict_risk(patients["Patient_C_Severe"]["data"])
            severe_threshold = patients["Patient_C_Severe"]["expected_min_risk"]
            
            print(f"📊 Patient C (Severe) Confidence Test:")
            print(f"   Risk: {severe_risk:.2f}%")
            print(f"   Threshold: > {severe_threshold}%")
            
            if severe_risk > severe_threshold:
                print("✅ SEVERE CONFIDENCE: PASSED")
            else:
                print("❌ SEVERE CONFIDENCE: FAILED")
                print("   ❌ Model not confident enough in severe case")
                confidence_passed = False
            print()
            
        except Exception as e:
            print(f"❌ ERROR testing severe confidence: {e}")
            confidence_passed = False
        
        overall_confidence = confidence_passed
        print("🔍 CONFIDENCE GAP SUMMARY:")
        if overall_confidence:
            print("✅ CONFIDENCE GAP TEST: PASSED")
            print("   ✅ Model shows appropriate confidence levels")
        else:
            print("❌ CONFIDENCE GAP TEST: FAILED")
            print("   ❌ Model under-confident or over-conservative")
        
        self.test_results["confidence_gap"] = {
            "passed": overall_confidence
        }
        
        return overall_confidence
    
    def test_stability(self) -> bool:
        """Test 3: Stability - Same input should produce same output"""
        print("\n" + "="*80)
        print("🧪 TEST 3: STABILITY CHECK")
        print("="*80)
        print("Testing: Same prediction 5 times should be identical")
        print()
        
        patients = self.create_synthetic_patients()
        borderline_patient = patients["Patient_B_Borderline"]["data"]
        
        predictions = []
        print(f"📊 Running 5 predictions on Patient B (Borderline)...")
        
        for i in range(5):
            try:
                risk_pct, risk_level, _ = self.predict_risk(borderline_patient)
                predictions.append(risk_pct)
                print(f"   Prediction {i+1}: {risk_pct:.2f}% ({risk_level})")
                time.sleep(0.1)  # Small delay between requests
            except Exception as e:
                print(f"❌ ERROR in prediction {i+1}: {e}")
                return False
        
        # Check if all predictions are identical
        first_prediction = predictions[0]
        all_identical = all(abs(pred - first_prediction) < 0.001 for pred in predictions)
        
        print()
        print("🔍 STABILITY ANALYSIS:")
        print(f"   Predictions: {[f'{p:.2f}%' for p in predictions]}")
        print(f"   Range: {min(predictions):.2f}% - {max(predictions):.2f}%")
        print(f"   Variance: {np.var(predictions):.6f}")
        
        if all_identical:
            print("✅ STABILITY TEST: PASSED")
            print("   ✅ Model produces consistent predictions")
        else:
            print("❌ STABILITY TEST: FAILED")
            print("   ❌ Model has random noise during inference")
            print("   ❌ This indicates dropout or randomness still active")
        
        self.test_results["stability"] = {
            "passed": all_identical,
            "predictions": predictions,
            "variance": np.var(predictions)
        }
        
        return all_identical
    
    def test_server_connectivity(self) -> bool:
        """Test server connectivity before running main tests"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final audit report"""
        print("\n" + "="*80)
        print("🎯 FINAL AUDIT REPORT")
        print("="*80)
        
        # Test Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["passed"])
        
        print(f"📊 OVERALL RESULTS: {passed_tests}/{total_tests} TESTS PASSED")
        print()
        
        # Individual Test Results
        print("📋 DETAILED RESULTS:")
        test_names = {
            "common_sense_ranking": "Common Sense Ranking",
            "confidence_gap": "Confidence Gap",
            "stability": "Stability Check"
        }
        
        for test_key, test_name in test_names.items():
            if test_key in self.test_results:
                status = "✅ PASSED" if self.test_results[test_key]["passed"] else "❌ FAILED"
                print(f"   {test_name}: {status}")
        
        print()
        
        # Risk Analysis Summary
        if "common_sense_ranking" in self.test_results:
            risks = self.test_results["common_sense_ranking"]["risks"]
            print("🎯 RISK CLASSIFICATION SUMMARY:")
            for patient_key, patient_data in risks.items():
                print(f"   {patient_data['full_result']['interpretation']}")
        
        print()
        
        # Final Verdict
        all_passed = passed_tests == total_tests
        if all_passed:
            print("🎉 FINAL VERDICT: MODEL ACCEPTED")
            print("   ✅ All critical tests passed")
            print("   ✅ Model is ready for production deployment")
            print("   ✅ Federated learning system is functioning correctly")
        else:
            print("❌ FINAL VERDICT: MODEL REJECTED")
            print("   ❌ Critical failures detected")
            print("   ❌ Model requires further tuning before deployment")
        
        print("="*80)
        
        return all_passed
    
    def run_full_audit(self) -> bool:
        """Run complete final acceptance test suite"""
        print("🚀 STARTING FINAL AUDIT FOR FEDERATED LEARNING MODEL")
        print(f"🌐 Target Server: {self.server_url}")
        print("="*80)
        
        # Check server connectivity
        if not self.test_server_connectivity():
            print("❌ FATAL: Cannot connect to server")
            print("   Ensure the server is running on", self.server_url)
            return False
        
        print("✅ Server connectivity verified")
        
        # Run all tests
        test_results = []
        
        # Test 1: Common Sense Ranking
        test_results.append(self.test_common_sense_ranking())
        
        # Test 2: Confidence Gap
        test_results.append(self.test_confidence_gap())
        
        # Test 3: Stability
        test_results.append(self.test_stability())
        
        # Generate final report
        all_passed = self.generate_final_report()
        
        return all_passed

def main():
    """Main execution function"""
    auditor = FinalAudit()
    
    try:
        success = auditor.run_full_audit()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during audit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
