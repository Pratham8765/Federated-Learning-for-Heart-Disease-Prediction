# setup_scaler.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_global_scaler():
    """Create and save a global scaler fitted on training data"""
    try:
        # Load the training dataset (same as clients use)
        df = pd.read_csv('diabetes_non_negative_part1_2000.csv')
        
        logger.info("Creating global scaler from training dataset...")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Define expected features
        feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Extract features
        if all(col in df.columns for col in feature_columns):
            X = df[feature_columns].values
        else:
            X = df.iloc[:, :8].values
            
        logger.info(f"Features shape: {X.shape}")
        
        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Save scaler to file
        joblib.dump(scaler, 'global_scaler.pkl')
        
        # Log scaler statistics for verification
        logger.info("Global scaler statistics:")
        logger.info(f"Mean: {scaler.mean_}")
        logger.info(f"Scale: {scaler.scale_}")
        
        print("✅ Global scaler created and saved to global_scaler.pkl")
        print(f"   - Fitted on {len(X)} samples")
        print(f"   - Feature means: {np.round(scaler.mean_, 4)}")
        print(f"   - Feature scales: {np.round(scaler.scale_, 4)}")
        
        return scaler
        
    except Exception as e:
        logger.error(f"Error creating global scaler: {e}")
        raise

def load_global_scaler():
    """Load the global scaler"""
    try:
        scaler = joblib.load('global_scaler.pkl')
        logger.info("✅ Global scaler loaded successfully")
        return scaler
    except FileNotFoundError:
        logger.error("❌ Global scaler not found. Run setup_scaler.py first.")
        return None
    except Exception as e:
        logger.error(f"Error loading global scaler: {e}")
        return None

def verify_scaler_consistency():
    """Verify that scaler is working correctly"""
    try:
        scaler = load_global_scaler()
        if scaler is None:
            return False
            
        # Test with sample data
        test_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # High risk case
        scaled_input = scaler.transform(test_input)
        
        print("🔍 Scaler Verification:")
        print(f"   Original input: {test_input[0]}")
        print(f"   Scaled input: {np.round(scaled_input[0], 4)}")
        
        # Check if scaling looks reasonable
        if np.all(np.abs(scaled_input) < 10):  # Most features should be within few standard deviations
            print("✅ Scaler verification passed")
            return True
        else:
            print("❌ Scaler verification failed - values too extreme")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying scaler: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up Global Scaler System")
    print("=" * 50)
    
    # Create global scaler
    scaler = create_global_scaler()
    
    # Verify consistency
    if verify_scaler_consistency():
        print("=" * 50)
        print("🎯 Global scaler setup completed successfully!")
        print("   - File: global_scaler.pkl")
        print("   - Ready for client and server use")
        print("=" * 50)
    else:
        print("=" * 50)
        print("❌ Scaler setup failed. Check logs for details.")
        print("=" * 50)
