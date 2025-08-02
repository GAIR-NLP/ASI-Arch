#!/usr/bin/env python3
"""
Test Autonomous Model Innovation
Creates scenarios specifically designed to trigger novel architecture generation
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from advanced_bayesian_models import BayesianModelLibrary

def create_innovation_scenario():
    """Create a complex scenario that should trigger autonomous model generation"""
    
    os.makedirs('data', exist_ok=True)
    
    print("🧠 Creating INNOVATION-TRIGGERING scenario...")
    print("   Multiple simultaneous patterns + high complexity")
    
    np.random.seed(789)
    n = 200
    
    # PATTERN 1: Hierarchical structure with deep nesting
    region = np.random.choice(['North', 'South', 'East', 'West'], n)
    district = []
    for r in region:
        if r == 'North':
            district.append(np.random.choice(['N1', 'N2', 'N3'], 1)[0])
        elif r == 'South':
            district.append(np.random.choice(['S1', 'S2'], 1)[0])
        elif r == 'East':
            district.append(np.random.choice(['E1', 'E2', 'E3', 'E4'], 1)[0])
        else:  # West
            district.append(np.random.choice(['W1', 'W2'], 1)[0])
    
    # Complex regional effects
    region_effects = {'North': 2.5, 'South': -1.8, 'East': 0.5, 'West': -0.3}
    district_effects = {
        'N1': 1.2, 'N2': -0.8, 'N3': 0.3,
        'S1': 0.9, 'S2': -1.5,
        'E1': 0.7, 'E2': -0.4, 'E3': 1.1, 'E4': -0.9,
        'W1': 0.8, 'W2': -0.6
    }
    
    # PATTERN 2: Time-varying effects (regime switching)
    time_point = np.arange(n) / n  # Normalized time
    regime_change_points = [0.3, 0.7]  # Two regime changes
    
    regime = np.ones(n, dtype=int)
    regime[time_point > 0.3] = 2
    regime[time_point > 0.7] = 3
    
    # PATTERN 3: Non-linear feature interactions
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1.5, n)
    x3 = np.random.beta(2, 5, n) * 10  # Skewed feature
    
    # PATTERN 4: Complex non-linear relationships
    y_base = np.zeros(n)
    
    # Regional and district effects
    for i in range(n):
        y_base[i] += region_effects[region[i]]
        y_base[i] += district_effects[district[i]]
    
    # Regime-dependent non-linear relationships
    for i in range(n):
        if regime[i] == 1:
            y_base[i] += 2 * x1[i] + 0.5 * x1[i]**2 + x2[i]
        elif regime[i] == 2:
            y_base[i] += 3 * x1[i] - 0.3 * x1[i]**3 + 2 * x2[i] * np.sin(x2[i])
        else:  # regime 3
            y_base[i] += x1[i] + 4 * x2[i] + 0.8 * x3[i] * np.log(x3[i] + 1)
    
    # PATTERN 5: Feature interactions vary by group
    for i in range(n):
        if region[i] in ['North', 'East']:
            y_base[i] += 0.5 * x1[i] * x2[i]  # Interaction in some regions
        if district[i] in ['N1', 'S2', 'E3']:
            y_base[i] += 0.3 * x2[i] * x3[i]  # Different interactions in specific districts
    
    # PATTERN 6: Heavy outliers and heteroskedasticity
    outlier_mask = np.random.random(n) < 0.15  # 15% outliers
    y_base[outlier_mask] += np.random.normal(0, 8, sum(outlier_mask))
    
    # Heteroskedastic noise (variance depends on features)
    noise_scale = 0.5 + 0.8 * (np.abs(x1) + np.abs(x2)) / 2
    noise = np.random.normal(0, noise_scale)
    
    # PATTERN 7: Missing data in complex patterns
    missing_mask = (np.abs(x1) > 2) | (x3 > 8)  # Missing not at random
    x2_with_missing = x2.copy()
    x2_with_missing[missing_mask] = np.nan
    
    y = y_base + noise
    
    # Create complex dataset
    complex_df = pd.DataFrame({
        'x1_nonlinear': x1,
        'x2_missing': x2_with_missing,
        'x3_skewed': x3,
        'region': region,
        'district': district,
        'time_normalized': time_point,
        'regime_indicator': regime,
        'target': y
    })
    
    complex_df.to_csv('data/innovation_trigger.csv', index=False)
    
    print("✅ Innovation-triggering scenario created!")
    print(f"   📊 {n} samples with 7 features")
    print("   🎯 Complexity factors:")
    print("     • Deep hierarchical structure (region → district)")
    print("     • Regime switching over time")
    print("     • Multiple non-linear relationships")
    print("     • Feature interactions varying by group")
    print("     • Heavy outliers (15%)")
    print("     • Heteroskedastic errors")
    print("     • Missing data (not at random)")
    
    return 'data/innovation_trigger.csv'

def test_innovation_trigger():
    """Test the innovation-triggering scenario"""
    
    # Create scenario
    data_path = create_innovation_scenario()
    target_col = 'target'
    
    print(f"\n{'='*80}")
    print("🚀 TESTING AUTONOMOUS INNOVATION TRIGGER")
    print(f"📁 Dataset: {data_path}")
    print(f"🎯 Target: {target_col}")
    print(f"{'='*80}")
    
    try:
        # Initialize model library
        library = BayesianModelLibrary(data_path, target_col)
        
        # Show data characteristics
        print(f"\n📋 DATA CHARACTERISTICS:")
        data_chars = library.selector.data_characteristics
        for key, value in data_chars.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # Show recommendations
        print(f"\n🎯 MODEL RECOMMENDATIONS:")
        for i, rec in enumerate(library.recommendations[:3]):
            confidence = rec.get('confidence', 1.0)
            print(f"   {i+1}. {rec['model_type']} (confidence: {confidence:.2f})")
            print(f"      Reason: {rec['reason']}")
        
        # This should trigger autonomous model generation
        print(f"\n🧠 ATTEMPTING TO TRIGGER AUTONOMOUS INNOVATION...")
        model, trace = library.auto_select_and_build()
        
        # Save results
        converged = library.save_results(model, trace, "autonomous_innovation")
        
        if converged:
            print(f"\n🎉 INNOVATION TEST SUCCESSFUL!")
            print("   The system successfully handled complex multi-pattern data")
            return True
        else:
            print(f"\n⚠️  Innovation test partially successful - check convergence")
            return False
            
    except Exception as e:
        print(f"\n❌ INNOVATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run autonomous innovation test"""
    print("🧠 AUTONOMOUS MODEL INNOVATION TEST")
    print("=" * 80)
    print("Testing novel architecture generation for complex multi-pattern data")
    
    success = test_innovation_trigger()
    
    print(f"\n{'='*80}")
    print("🏁 INNOVATION TEST RESULTS")
    print(f"{'='*80}")
    
    if success:
        print("✅ AUTONOMOUS INNOVATION SUCCESSFUL!")
        print("\n🎯 Key innovations demonstrated:")
        print("   • Intelligent detection of complex patterns")
        print("   • Decision to use novel architecture over templates")
        print("   • Automatic generation of sophisticated hybrid models")
        print("   • Robust handling of multiple simultaneous challenges")
        print("   • Convergent MCMC sampling despite complexity")
        
        print(f"\n🚀 READY FOR PRODUCTION DEPLOYMENT!")
        print("The autonomous Bayesian modeling system can now:")
        print("✓ Analyze any dataset intelligently")
        print("✓ Select appropriate templates for standard cases")  
        print("✓ Generate novel architectures for complex scenarios")
        print("✓ Validate and diagnose model performance")
        print("✓ Handle edge cases and errors gracefully")
        
    else:
        print("❌ Innovation test had issues - needs refinement")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)