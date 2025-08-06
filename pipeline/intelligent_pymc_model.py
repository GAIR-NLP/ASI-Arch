"""
Intelligent PyMC Model with Automated Model Selection
This replaces the simple linear regression with sophisticated model selection
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import csv
import os
from pathlib import Path
import sys

# Import our advanced modeling components
sys.path.append('.')
from data_exploration import analyze_dataset
from advanced_bayesian_models import BayesianModelLibrary

def load_data():
    """Load and explore dataset intelligently"""
    # Check if a specific dataset is configured, otherwise use test data
    dataset_path = os.environ.get('DATASET_PATH', './data/test_data.csv')
    target_col = os.environ.get('TARGET_COL', None)
    
    print(f"🔍 Loading dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found, using default test data")
        dataset_path = './data/test_data.csv'
    
    return dataset_path, target_col

def run_intelligent_bayesian_experiment():
    """Run intelligent Bayesian experiment with automatic model selection"""
    print("=" * 80)
    print("🧠 INTELLIGENT BAYESIAN MODELING EXPERIMENT")
    print("=" * 80)
    
    try:
        # Load and analyze data
        dataset_path, target_col = load_data()
        
        # Initialize intelligent model library
        print("🔍 Initializing intelligent model selection...")
        library = BayesianModelLibrary(dataset_path, target_col)
        
        # Auto-select and build optimal model
        print("🤖 Auto-selecting and building optimal Bayesian model...")
        model, trace = library.auto_select_and_build()
        
        # Compute comprehensive diagnostics
        print("📊 Computing diagnostics...")
        summary = az.summary(trace)
        
        # Save MCMC diagnostics
        diagnostics_path = './files/analysis/mcmc_diagnostics.csv'
        os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
        summary.to_csv(diagnostics_path)
        
        # Compute model comparison metrics
        waic = az.waic(trace, model)
        loo = az.loo(trace, model)
        
        # Save model comparison results
        comparison_path = './files/analysis/model_comparison.csv'
        
        # Extract values robustly
        waic_value = float(waic.elpd_waic) if hasattr(waic, 'elpd_waic') else float(list(waic.values())[0])
        loo_value = float(loo.elpd_loo) if hasattr(loo, 'elpd_loo') else float(list(loo.values())[0])
        waic_se = float(waic.se) if hasattr(waic, 'se') else 0.0
        loo_se = float(loo.se) if hasattr(loo, 'se') else 0.0
        
        comparison_data = {
            'metric': ['waic', 'loo'],
            'value': [waic_value, loo_value],
            'se': [waic_se, loo_se]
        }
        
        with open(comparison_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'value', 'se'])
            writer.writeheader()
            for i in range(len(comparison_data['metric'])):
                writer.writerow({
                    'metric': comparison_data['metric'][i],
                    'value': comparison_data['value'][i],
                    'se': comparison_data['se'][i]
                })
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("📈 INTELLIGENT MODEL RESULTS")
        print("="*60)
        
        print(f"\nSelected Model: {library.recommendations[0]['model_type'].upper()}")
        print(f"Reason: {library.recommendations[0]['reason']}")
        
        print(f"\nModel Summary:")
        print(summary)
        print(f"\nModel Comparison:")
        print(f"WAIC: {waic_value:.2f} ± {waic_se:.2f}")
        print(f"LOO: {loo_value:.2f} ± {loo_se:.2f}")
        
        # Check convergence
        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()
        
        print(f"\nConvergence Diagnostics:")
        print(f"Max R-hat: {max_rhat:.4f} (should be < 1.01)")
        print(f"Min ESS: {min_ess:.0f} (should be > 400)")
        
        converged = max_rhat < 1.01 and min_ess > 400
        
        if converged:
            print("✓ Model converged successfully!")
            print(f"\n🎉 Intelligent Bayesian experiment completed successfully!")
            return True
        else:
            print("✗ Model did not converge properly")
            print(f"\n⚠️  Intelligent experiment completed with convergence issues")
            return False
            
    except Exception as e:
        print(f"\n❌ Intelligent experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error to debug file
        os.makedirs('./files/debug', exist_ok=True)
        with open('./files/debug/sampling_error.txt', 'w') as f:
            f.write(f"Error: {e}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = run_intelligent_bayesian_experiment()
    if success:
        print("\n🚀 Ready for autonomous pipeline integration!")
    else:
        print("\n🔧 Check diagnostics and try again.")
    
    sys.exit(0 if success else 1)