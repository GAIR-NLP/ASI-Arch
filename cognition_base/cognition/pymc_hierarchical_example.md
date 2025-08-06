# Hierarchical Linear Regression with PyMC

This notebook demonstrates hierarchical linear regression using PyMC for grouped data analysis.

## Background

Hierarchical models are useful when dealing with grouped data where we expect some similarity within groups but also want to account for group-level differences. This approach provides a principled way to "borrow strength" across groups while still allowing for group-specific parameters.

## Data Setup

We'll simulate data with a hierarchical structure where different groups have different intercepts but similar slopes.

```python
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulate hierarchical data
n_groups = 8
n_per_group = 15
n_total = n_groups * n_per_group

# Group-specific intercepts
true_group_intercepts = np.random.normal(2, 1, n_groups)
true_slope = 1.5
true_sigma = 0.5

# Generate data
group_idx = np.repeat(np.arange(n_groups), n_per_group)
x = np.random.normal(0, 1, n_total)
y = true_group_intercepts[group_idx] + true_slope * x + np.random.normal(0, true_sigma, n_total)

# Create DataFrame
df = pd.DataFrame({
    'y': y,
    'x': x,
    'group': group_idx
})
```

## Hierarchical Model

We'll build a hierarchical model where group intercepts are drawn from a common population distribution:

```python
with pm.Model() as hierarchical_model:
    # Hyperpriors for population-level parameters
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=5)
    
    # Group-specific intercepts
    alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
    
    # Common slope
    beta = pm.Normal('beta', mu=0, sigma=5)
    
    # Model error
    sigma = pm.HalfNormal('sigma', sigma=2)
    
    # Linear predictor
    mu = alpha[group_idx] + beta * x
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
    
    # Sample from posterior
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, random_seed=42)
```

## Model Diagnostics

Always check convergence and model fit:

```python
# Summary statistics
summary = az.summary(trace)
print(summary)

# Check convergence
max_rhat = summary['r_hat'].max()
min_ess = summary['ess_bulk'].min()

print(f"Max R-hat: {max_rhat:.4f} (should be < 1.01)")
print(f"Min ESS: {min_ess:.0f} (should be > 400)")

# Trace plots
az.plot_trace(trace, var_names=['mu_alpha', 'sigma_alpha', 'beta', 'sigma'])
plt.tight_layout()
plt.show()

# Posterior predictive checks
with hierarchical_model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model))
plt.show()
```

## Model Comparison

Compare with a pooled model to demonstrate the benefits of hierarchical modeling:

```python
# Pooled model (ignores group structure)
with pm.Model() as pooled_model:
    alpha_pooled = pm.Normal('alpha_pooled', mu=0, sigma=10)
    beta_pooled = pm.Normal('beta_pooled', mu=0, sigma=5)
    sigma_pooled = pm.HalfNormal('sigma_pooled', sigma=2)
    
    mu_pooled = alpha_pooled + beta_pooled * x
    y_obs_pooled = pm.Normal('y_obs_pooled', mu=mu_pooled, sigma=sigma_pooled, observed=y)
    
    trace_pooled = pm.sample(2000, tune=1000, chains=4, random_seed=42, 
                            idata_kwargs={'log_likelihood': True})

# Compare models using WAIC
with hierarchical_model:
    trace_hier = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                          idata_kwargs={'log_likelihood': True})

comparison = az.compare({
    'Hierarchical': trace_hier,
    'Pooled': trace_pooled
})

print("Model Comparison:")
print(comparison)
```

## Key Insights

1. **Partial Pooling**: The hierarchical model automatically balances between complete pooling (all groups identical) and no pooling (groups completely independent).

2. **Shrinkage**: Group-specific estimates are "shrunk" toward the population mean, with the amount of shrinkage depending on the group sample size and population variability.

3. **Uncertainty Quantification**: The model provides proper uncertainty quantification at both the group and population levels.

4. **Predictive Performance**: Hierarchical models often have better out-of-sample predictive performance, especially for new groups.

## Extensions

This basic framework can be extended to include:
- Hierarchical slopes in addition to intercepts
- Multiple grouping factors (crossed or nested)
- Non-linear relationships
- Different likelihood families (Poisson, binomial, etc.)
- Time-varying parameters

The key is to think about the structure in your data and build that structure into your model through appropriate hierarchical components.