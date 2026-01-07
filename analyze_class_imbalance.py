# analyze_class_imbalance.py
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_class_distribution():
    """Analyze class imbalance in the diabetes dataset"""
    try:
        # Load the training dataset
        df = pd.read_csv('diabetes_non_negative_part1_2000.csv')
        
        print("🔍 CLASS IMBALANCE ANALYSIS")
        print("=" * 50)
        
        # Extract target variable
        if 'Outcome' in df.columns:
            y = df['Outcome'].values
        else:
            y = df.iloc[:, -1].values
            
        # Convert to binary if needed
        if len(np.unique(y)) > 2:
            y = (y > 0).astype(int)
        
        # Count classes
        class_counts = Counter(y)
        total_samples = len(y)
        
        print(f"📊 Dataset: {total_samples} samples")
        print(f"   Class 0 (No Diabetes): {class_counts[0]} samples ({class_counts[0]/total_samples*100:.1f}%)")
        print(f"   Class 1 (Diabetes): {class_counts[1]} samples ({class_counts[1]/total_samples*100:.1f}%)")
        
        # Calculate imbalance ratio
        imbalance_ratio = class_counts[0] / class_counts[1]
        print(f"\n⚖️  Imbalance Ratio (Class0/Class1): {imbalance_ratio:.2f}:1")
        
        # Calculate class weights for balanced loss
        # Formula: weight_class = total_samples / (num_classes * count_class)
        weight_class_0 = total_samples / (2 * class_counts[0])
        weight_class_1 = total_samples / (2 * class_counts[1])
        
        print(f"\n🎯 RECOMMENDED CLASS WEIGHTS:")
        print(f"   Weight for Class 0 (No Diabetes): {weight_class_0:.3f}")
        print(f"   Weight for Class 1 (Diabetes): {weight_class_1:.3f}")
        print(f"   Weight Ratio (Class1/Class0): {weight_class_1/weight_class_0:.2f}")
        
        # PyTorch specific recommendation
        pos_weight = class_counts[0] / class_counts[1]  # For BCEWithLogitsLoss
        print(f"\n🔧 PYTORCH BCEWithLogitsLoss pos_weight: {pos_weight:.2f}")
        
        # Analyze Non-IID splits
        print(f"\n📊 NON-IID SPLIT ANALYSIS:")
        
        # Client 1 (first 60%)
        split_idx = int(0.6 * len(df))
        y_client1 = y[:split_idx]
        client1_counts = Counter(y_client1)
        
        # Client 2 (last 40%)
        y_client2 = y[split_idx:]
        client2_counts = Counter(y_client2)
        
        print(f"   Client 1 (60% data): {len(y_client1)} samples")
        print(f"     Class 0: {client1_counts[0]} ({client1_counts[0]/len(y_client1)*100:.1f}%)")
        print(f"     Class 1: {client1_counts[1]} ({client1_counts[1]/len(y_client1)*100:.1f}%)")
        print(f"     Imbalance: {client1_counts[0]/client1_counts[1]:.2f}:1")
        
        print(f"   Client 2 (40% data): {len(y_client2)} samples")
        print(f"     Class 0: {client2_counts[0]} ({client2_counts[0]/len(y_client2)*100:.1f}%)")
        print(f"     Class 1: {client2_counts[1]} ({client2_counts[1]/len(y_client2)*100:.1f}%)")
        print(f"     Imbalance: {client2_counts[0]/client2_counts[1]:.2f}:1")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if imbalance_ratio > 2.0:
            print("   ✅ SEVERE IMBALANCE DETECTED - Class weights ESSENTIAL")
            print("   ✅ Use pos_weight = {pos_weight:.2f} in BCEWithLogitsLoss")
        elif imbalance_ratio > 1.5:
            print("   ✅ MODERATE IMBALANCE - Class weights RECOMMENDED")
        else:
            print("   ⚠️  Mild imbalance - Class weights optional")
            
        print("   ✅ Consider reducing regularization (Dropout: 0.2, Weight Decay: 1e-5)")
        print("   ✅ Increase local_epochs to 3-5 for better learning")
        
        return {
            'total_samples': total_samples,
            'class_counts': class_counts,
            'imbalance_ratio': imbalance_ratio,
            'pos_weight': pos_weight,
            'class_weights': {0: weight_class_0, 1: weight_class_1}
        }
        
    except Exception as e:
        print(f"❌ Error analyzing class distribution: {e}")
        return None

if __name__ == "__main__":
    results = analyze_class_distribution()
    
    if results:
        print(f"\n🎯 SUMMARY:")
        print(f"   - Dataset has {results['imbalance_ratio']:.1f}:1 class imbalance")
        print(f"   - Use pos_weight={results['pos_weight']:.2f} for balanced training")
        print(f"   - This should significantly improve high-risk prediction confidence")
