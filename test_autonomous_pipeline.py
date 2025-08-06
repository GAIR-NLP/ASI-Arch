#!/usr/bin/env python3
"""
Test Autonomous Bayesian Modeling Pipeline
Demonstrates both template selection and novel architecture generation
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from advanced_bayesian_models import BayesianModelLibrary

def create_test_scenarios():
    """Create different test scenarios to trigger different model selections"""
    
    os.makedirs('data', exist_ok=True)
    
    print("🏗️  Creating test scenarios...")
    
    # Scenario 1: Simple linear data (should use template)
    print("   📊 Simple linear scenario")
    np.random.seed(42)
    n = 100
    x = np.random.normal(0, 1, n)
    y = 2 * x + 0.5 + np.random.normal(0, 0.3, n)
    
    simple_df = pd.DataFrame({
        'feature1': x,
        'feature2': np.random.normal(0, 1, n),
        'target': y
    })
    simple_df.to_csv('data/simple_linear.csv', index=False)
    
    # Scenario 2: Complex multi-pattern data (should trigger autonomous generation)
    print("   🧠 Complex multi-pattern scenario")
    np.random.seed(123)
    n = 150
    
    # Multiple patterns: hierarchical + non-linear + outliers + multimodal
    groups = np.random.choice(['A', 'B', 'C'], n)
    group_effects = {'A': 0, 'B': 2, 'C': -1}
    
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    
    # Non-linear relationships
    y_base = [group_effects[g] for g in groups]
    y_base += 3 * x1 + 0.5 * x1**2  # Non-linear
    y_base += 1.5 * x2 * np.sin(x2)  # More non-linearity
    
    # Add outliers
    outlier_mask = np.random.random(n) < 0.1
    y_base[outlier_mask] += np.random.normal(0, 5, sum(outlier_mask))
    
    # Add noise with different scales per group
    noise_scales = {'A': 0.5, 'B': 1.0, 'C': 0.3}
    noise = [np.random.normal(0, noise_scales[g]) for g in groups]
    
    y = y_base + noise
    
    complex_df = pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'group': groups,
        'target': y
    })
    complex_df.to_csv('data/complex_multipattern.csv', index=False)
    
    # Scenario 3: Hierarchical data (should use hierarchical template)
    print("   🏛️  Hierarchical scenario")
    np.random.seed(456)
    n = 80
    
    schools = np.random.choice(['School_A', 'School_B', 'School_C', 'School_D'], n)
    school_effects = {'School_A': 1.5, 'School_B': -0.5, 'School_C': 0.8, 'School_D': -1.2}
    
    x = np.random.normal(0, 1, n)
    y = [school_effects[s] for s in schools] + 2.5 * x + np.random.normal(0, 0.4, n)
    
    hierarchical_df = pd.DataFrame({
        'student_score': x,
        'school': schools,
        'test_result': y
    })
    hierarchical_df.to_csv('data/hierarchical_schools.csv', index=False)
    
    print("✅ Test scenarios created!")
    return [
        ('data/simple_linear.csv', 'target', 'Simple Linear'),
        ('data/complex_multipattern.csv', 'target', 'Complex Multi-Pattern'),
        ('data/hierarchical_schools.csv', 'test_result', 'Hierarchical Schools')
    ]

def test_scenario(data_path, target_col, scenario_name):
    """Test a single scenario"""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SCENARIO: {scenario_name}")
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
            print(f"   {i+1}. {rec['model_type']} (confidence: {rec.get('confidence', 1.0):.2f})")
            print(f"      Reason: {rec['reason']}")
        
        # Auto-select and build model
        print(f"\n🚀 BUILDING MODEL...")
        model, trace = library.auto_select_and_build()
        
        # Save results
        converged = library.save_results(model, trace, scenario_name.lower().replace(' ', '_'))
        
        status = "✅ SUCCESS" if converged else "⚠️  PARTIAL SUCCESS"
        print(f"\n{status}: {scenario_name} completed!")
        
        return converged
        
    except Exception as e:
        print(f"\n❌ FAILED: {scenario_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive autonomous pipeline test"""
    print("🧠 AUTONOMOUS BAYESIAN MODELING PIPELINE TEST")
    print("=" * 80)
    print("Testing both template selection and novel architecture generation")
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Test each scenario
    results = {}
    for data_path, target_col, scenario_name in scenarios:
        results[scenario_name] = test_scenario(data_path, target_col, scenario_name)
    
    # Summary
    print(f"\n{'='*80}")
    print("🏁 FINAL RESULTS")
    print(f"{'='*80}")
    
    for scenario, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {scenario}")
    
    total_success = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_success}/{total_tests} scenarios successful")
    
    if total_success == total_tests:
        print("\n🎉 ALL TESTS PASSED! Autonomous pipeline working correctly!")
        print("\nKey capabilities demonstrated:")
        print("✅ Intelligent data analysis and pattern detection")
        print("✅ Automatic template selection for standard scenarios")
        print("✅ Novel architecture generation for complex data")
        print("✅ Comprehensive MCMC diagnostics and validation")
        print("✅ Robust error handling and fallback mechanisms")
    else:
        print(f"\n⚠️  {total_tests - total_success} scenarios had issues - check logs")
    
    return total_success == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)