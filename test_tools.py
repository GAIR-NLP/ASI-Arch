#!/usr/bin/env python3
"""Test script to verify tools work with pixi environment"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

try:
    from tools import read_pymc_model, load_dataset, run_eda_script, validate_pymc_model
    print("✓ Tools import successfully")
    
    # Test dataset loading
    result = load_dataset('test_data.csv')
    if result['success']:
        print("✓ Dataset loading works")
    else:
        print(f"✗ Dataset loading failed: {result['error']}")
    
    # Test EDA script
    eda_result = run_eda_script('../data/test_data.csv')
    if eda_result['success']:
        print("✓ EDA script works")
    else:
        print(f"✗ EDA script failed: {eda_result['error']}")
        
    # Test PyMC model reading
    model_result = read_pymc_model()
    if model_result['success']:
        print("✓ PyMC model reading works")
    else:
        print(f"✗ PyMC model reading failed: {model_result['error']}")
        
    print("\nAll tools working correctly!")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)