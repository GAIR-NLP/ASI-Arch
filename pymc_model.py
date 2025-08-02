"""
Test PyMC Model - Simple Linear Regression
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import csv
import os
from pathlib import Path

def load_data():
    """Load test dataset"""
    df = pd.read_csv('./data/test_data.csv')
    return df['x'].values, df['y'].values

def run_bayesian_model():
    """Run a simple Bayesian linear regression model"""
    print("Loading data...")
    x, y = load_data()
    
    print("Building PyMC model...")
    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Expected value of outcome
        mu = alpha + beta * x
        
        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
        
        print("Running MCMC sampling...")
        # Sample from the posterior with log likelihood computation
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.8, 
                         random_seed=42, progressbar=True, idata_kwargs={'log_likelihood': True})
        
        print("Computing diagnostics...")
        # Compute convergence diagnostics
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
        
        # Extract values from WAIC and LOO objects
        waic_value = float(waic.elpd_waic) if hasattr(waic, 'elpd_waic') else float(waic.values()[0])
        loo_value = float(loo.elpd_loo) if hasattr(loo, 'elpd_loo') else float(loo.values()[0])
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
        
        # Print summary statistics
        print("\nModel Summary:")
        print(summary)
        print(f"\nWAIC: {waic_value:.2f} ± {waic_se:.2f}")
        print(f"LOO: {loo_value:.2f} ± {loo_se:.2f}")
        
        # Check convergence
        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()
        
        print(f"\nConvergence Diagnostics:")
        print(f"Max R-hat: {max_rhat:.4f} (should be < 1.01)")
        print(f"Min ESS: {min_ess:.0f} (should be > 400)")
        
        if max_rhat < 1.01 and min_ess > 400:
            print("✓ Model converged successfully!")
            return True
        else:
            print("✗ Model did not converge properly")
            return False

if __name__ == "__main__":
    try:
        success = run_bayesian_model()
        if success:
            print("\nBayesian experiment completed successfully!")
        else:
            print("\nBayesian experiment completed with convergence issues")
    except Exception as e:
        print(f"\nError running Bayesian model: {e}")
        # Save error to debug file
        os.makedirs('./files/debug', exist_ok=True)
        with open('./files/debug/sampling_error.txt', 'w') as f:
            f.write(f"Error: {e}\n")
        raise