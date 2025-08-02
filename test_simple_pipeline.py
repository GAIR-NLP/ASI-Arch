#!/usr/bin/env python3
"""Simple test of the pipeline components without the agents framework"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

# Add pipeline to path
sys.path.append('pipeline')

try:
    # Test basic imports
    from config import Config
    print("✓ Config imported successfully")
    print(f"Source file: {Config.SOURCE_FILE}")
    print(f"Bash script: {Config.BASH_SCRIPT}")
    
    # Test if files exist
    source_path = Path(Config.SOURCE_FILE)
    script_path = Path(Config.BASH_SCRIPT)
    
    if source_path.exists():
        print("✓ PyMC model file exists")
    else:
        print("✗ PyMC model file missing")
    
    if script_path.exists():
        print("✓ Experiment script exists")
    else:
        print("✗ Experiment script missing")
    
    # Test running the experiment script directly
    print("\nTesting experiment script...")
    try:
        # Make script executable and run it
        os.chmod(script_path, 0o755)
        result = subprocess.run([str(script_path.absolute()), "pipeline_test"], 
                               capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ Experiment script runs successfully")
        else:
            print(f"✗ Experiment script failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Experiment script timed out")
    except Exception as e:
        print(f"✗ Error running experiment script: {e}")
    
    # Check if results were generated
    results_dir = Path("files/analysis")
    if results_dir.exists():
        files = list(results_dir.glob("*.csv"))
        print(f"✓ Found {len(files)} result files: {[f.name for f in files]}")
    else:
        print("✗ No results directory found")
        
    print("\nPipeline test completed!")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)