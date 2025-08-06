"""
Advanced Bayesian Models for Complex Data Structures
"""
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import csv
import os
from pathlib import Path
from data_exploration import analyze_dataset

class BayesianModelLibrary:
    """Library of sophisticated Bayesian models"""
    
    def __init__(self, data_path, target_col=None):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.target_col = target_col or self.df.columns[-1]
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]
        
        # Run data exploration
        print("🔍 Analyzing dataset for model selection...")
        self.selector, self.recommendations = analyze_dataset(data_path, target_col)
        
    def build_hierarchical_model(self):
        """Build hierarchical Bayesian model for grouped data"""
        print("\n🏗️  Building Hierarchical Bayesian Model...")
        
        # Identify grouping variables
        categorical_cols = [col for col in self.df.columns 
                           if self.df[col].dtype == 'object' and col != self.target_col]
        
        if not categorical_cols:
            raise ValueError("No categorical variables found for hierarchical modeling")
        
        # Use first categorical variable as main group
        group_col = categorical_cols[0]
        groups = self.df[group_col].unique()
        n_groups = len(groups)
        
        # Prepare data
        y = self.df[self.target_col].values
        group_idx = pd.Categorical(self.df[group_col]).codes
        
        # Get numeric predictors
        numeric_cols = [col for col in self.feature_cols 
                       if self.df[col].dtype in ['int64', 'float64']]
        
        with pm.Model() as hierarchical_model:
            # Hyperpriors for group-level parameters
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=5)
            
            # Group-specific intercepts
            alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
            
            if numeric_cols:
                # Hierarchical slopes for numeric predictors
                X = self.df[numeric_cols].values
                n_predictors = X.shape[1]
                
                mu_beta = pm.Normal('mu_beta', mu=0, sigma=5, shape=n_predictors)
                sigma_beta = pm.HalfNormal('sigma_beta', sigma=2, shape=n_predictors)
                
                beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, 
                               shape=(n_groups, n_predictors))
                
                # Linear predictor
                mu = alpha[group_idx] + pm.math.sum(beta[group_idx] * X, axis=1)
            else:
                mu = alpha[group_idx]
            
            # Likelihood
            sigma = pm.HalfNormal('sigma', sigma=5)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=42, idata_kwargs={'log_likelihood': True})
        
        return hierarchical_model, trace
    
    def build_mixture_model(self, n_components=2):
        """Build Gaussian mixture model for multimodal data"""
        print(f"\n🧩 Building Gaussian Mixture Model ({n_components} components)...")
        
        y = self.df[self.target_col].values
        n_obs = len(y)
        
        with pm.Model() as mixture_model:
            # Mixture weights
            weights = pm.Dirichlet('weights', a=np.ones(n_components))
            
            # Component parameters
            mu = pm.Normal('mu', mu=y.mean(), sigma=y.std(), shape=n_components)
            sigma = pm.HalfNormal('sigma', sigma=y.std(), shape=n_components)
            
            # Mixture likelihood
            mixture = pm.NormalMixture('mixture', w=weights, mu=mu, sigma=sigma, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95,
                            random_seed=42, idata_kwargs={'log_likelihood': True})
        
        return mixture_model, trace
    
    def build_poisson_regression(self):
        """Build Poisson regression for count data"""
        print("\n📊 Building Poisson Regression Model...")
        
        y = self.df[self.target_col].values
        
        # Get numeric predictors
        numeric_cols = [col for col in self.feature_cols 
                       if self.df[col].dtype in ['int64', 'float64']]
        
        if not numeric_cols:
            raise ValueError("No numeric predictors found for Poisson regression")
        
        X = self.df[numeric_cols].values
        n_predictors = X.shape[1]
        
        with pm.Model() as poisson_model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=5)
            beta = pm.Normal('beta', mu=0, sigma=2, shape=n_predictors)
            
            # Linear predictor (log link)
            eta = alpha + pm.math.dot(X, beta)
            mu = pm.math.exp(eta)
            
            # Likelihood
            y_obs = pm.Poisson('y_obs', mu=mu, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=42, idata_kwargs={'log_likelihood': True})
        
        return poisson_model, trace
    
    def build_robust_regression(self):
        """Build robust regression with Student-t likelihood"""
        print("\n💪 Building Robust Regression Model...")
        
        y = self.df[self.target_col].values
        
        # Get numeric predictors
        numeric_cols = [col for col in self.feature_cols 
                       if self.df[col].dtype in ['int64', 'float64']]
        
        if not numeric_cols:
            raise ValueError("No numeric predictors found for robust regression")
        
        X = self.df[numeric_cols].values
        n_predictors = X.shape[1]
        
        with pm.Model() as robust_model:
            # Priors
            alpha = pm.Normal('alpha', mu=y.mean(), sigma=y.std())
            beta = pm.Normal('beta', mu=0, sigma=2, shape=n_predictors)
            sigma = pm.HalfNormal('sigma', sigma=y.std())
            nu = pm.Exponential('nu', lam=1/10)  # Degrees of freedom for t-distribution
            
            # Linear predictor
            mu = alpha + pm.math.dot(X, beta)
            
            # Robust likelihood (Student-t)
            y_obs = pm.StudentT('y_obs', nu=nu, mu=mu, sigma=sigma, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=42, idata_kwargs={'log_likelihood': True})
        
        return robust_model, trace
    
    def build_polynomial_regression(self, degree=2):
        """Build polynomial regression for non-linear relationships"""
        print(f"\n📈 Building Polynomial Regression Model (degree {degree})...")
        
        y = self.df[self.target_col].values
        
        # Get numeric predictors
        numeric_cols = [col for col in self.feature_cols 
                       if self.df[col].dtype in ['int64', 'float64']]
        
        if not numeric_cols:
            raise ValueError("No numeric predictors found for polynomial regression")
        
        # Use first numeric column for polynomial features
        x = self.df[numeric_cols[0]].values
        
        # Create polynomial features
        X_poly = np.column_stack([x**i for i in range(1, degree + 1)])
        
        with pm.Model() as poly_model:
            # Priors
            alpha = pm.Normal('alpha', mu=y.mean(), sigma=y.std())
            beta = pm.Normal('beta', mu=0, sigma=2, shape=degree)
            sigma = pm.HalfNormal('sigma', sigma=y.std())
            
            # Polynomial predictor
            mu = alpha + pm.math.dot(X_poly, beta)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Sample
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9,
                            random_seed=42, idata_kwargs={'log_likelihood': True})
        
        return poly_model, trace
    
    def auto_select_and_build(self):
        """Automatically select and build the most appropriate model"""
        print("\n🤖 AUTO-SELECTING OPTIMAL BAYESIAN MODEL...")
        
        if not self.recommendations:
            raise ValueError("No model recommendations available")
        
        # Check if we should generate a novel architecture
        should_innovate = self._should_generate_novel_architecture()
        
        if should_innovate:
            print("🧠 GENERATING NOVEL ARCHITECTURE - Data requires innovation!")
            return self._generate_autonomous_model()
        
        # Get top recommendation from predefined templates
        top_rec = self.recommendations[0]
        model_type = top_rec['model_type']
        
        print(f"Selected template: {model_type}")
        print(f"Reason: {top_rec['reason']}")
        
        try:
            if model_type == 'hierarchical_model':
                return self.build_hierarchical_model()
            elif model_type == 'poisson_regression':
                return self.build_poisson_regression()
            elif model_type == 'polynomial_regression':
                return self.build_polynomial_regression()
            elif model_type == 'mixture_model':
                return self.build_mixture_model()
            else:
                # Default to robust regression
                print("Falling back to robust regression...")
                return self.build_robust_regression()
                
        except Exception as e:
            print(f"Failed to build {model_type}: {e}")
            print("Falling back to robust regression...")
            return self.build_robust_regression()
    
    def _should_generate_novel_architecture(self):
        """Determine if we should generate a novel architecture instead of using templates"""
        # Innovation triggers based on data characteristics
        data_chars = self.selector.data_characteristics
        patterns = data_chars.get('patterns', {})
        
        complexity_score = data_chars.get('complexity', {}).get('overall_score', 0)
        n_patterns = sum([
            patterns.get('hierarchical_structure', False),
            patterns.get('multimodal', False), 
            patterns.get('temporal', False),
            len(patterns.get('nonlinear_relationships', [])) > 0,
            data_chars.get('missing_pct', 0) > 5  # More than 5% missing
        ])
        
        # Force innovation for demonstration if we have multiple complex patterns
        force_innovation = (
            patterns.get('hierarchical_structure', False) and 
            patterns.get('temporal', False) and
            patterns.get('multimodal', False) and
            complexity_score >= 5
        )
        
        # Generate novel architecture if:
        innovation_needed = (
            force_innovation or              # Multiple complex patterns
            complexity_score > 7 or          # High complexity
            n_patterns >= 3 or               # Multiple simultaneous patterns  
            data_chars.get('target_distribution', {}).get('likely_distribution') == 'unknown' or
            self.recommendations[0].get('confidence', 1.0) < 0.6  # Low template confidence
        )
        
        if innovation_needed:
            print(f"🎯 Innovation triggers detected:")
            print(f"   Complexity: {complexity_score}/10")
            print(f"   Patterns: {n_patterns}")
            print(f"   Hierarchical: {patterns.get('hierarchical_structure', False)}")
            print(f"   Temporal: {patterns.get('temporal', False)}")
            print(f"   Multimodal: {patterns.get('multimodal', False)}")
            print(f"   Force innovation: {force_innovation}")
            print(f"   Template confidence: {self.recommendations[0].get('confidence', 1.0):.2f}")
        
        return innovation_needed
    
    def _generate_autonomous_model(self):
        """Generate a completely novel PyMC model architecture"""
        print("🚀 AUTONOMOUS MODEL GENERATION INITIATED")
        print("=" * 60)
        
        # In full implementation, this would call the autonomous generator
        # For now, demonstrate with a sophisticated hybrid model
        return self._build_autonomous_hybrid_model()
    
    def _build_autonomous_hybrid_model(self):
        """Build a sophisticated hybrid model as autonomous generation demo"""
        print("🛠️  Building autonomous hybrid architecture...")
        print("   Innovations: Regime switching + adaptive importance + robust mixtures")
        
        y = self.df[self.target_col].values
        
        # Get numeric predictors
        numeric_cols = [col for col in self.feature_cols 
                       if self.df[col].dtype in ['int64', 'float64']]
        
        if not numeric_cols:
            # Fallback for datasets without numeric predictors
            return self.build_robust_regression()
        
        X = self.df[numeric_cols].values
        n_features = X.shape[1]
        
        with pm.Model() as autonomous_model:
            print("   📐 Building regime-switching foundation...")
            
            # INNOVATION 1: Adaptive feature importance learning
            global_importance = pm.Dirichlet('global_importance', a=np.ones(n_features))
            
            # INNOVATION 2: Regime-switching model with learned transitions
            if len(y) > 10:  # Only for sufficient data
                regime_logits = pm.Normal('regime_logits', 0, 1, shape=len(y))
                regime_probs = pm.Deterministic('regime_probs', pm.math.sigmoid(regime_logits))
                regime = pm.Bernoulli('regime', p=regime_probs, shape=len(y))
                
                # Regime-specific parameters
                alpha_0 = pm.Normal('alpha_0', y.mean(), y.std())
                alpha_1 = pm.Normal('alpha_1', y.mean(), y.std())
                
                beta_0 = pm.Normal('beta_0', 0, global_importance)
                beta_1 = pm.Normal('beta_1', 0, global_importance)
                
                # INNOVATION 3: Non-linear feature transformations
                transform_power = pm.Normal('transform_power', 1, 0.3, shape=n_features)
                X_transformed = pm.math.exp(transform_power[None, :] * pm.math.log(pm.math.abs(X) + 1e-6))
                
                # Regime-dependent predictions
                mu_0 = alpha_0 + pm.math.sum(beta_0 * X_transformed, axis=1)
                mu_1 = alpha_1 + pm.math.sum(beta_1 * X_transformed, axis=1)
                mu = regime * mu_1 + (1 - regime) * mu_0
                
                # INNOVATION 4: Adaptive error model per regime
                sigma_0 = pm.HalfNormal('sigma_0', y.std())
                sigma_1 = pm.HalfNormal('sigma_1', y.std())
                sigma = regime * sigma_1 + (1 - regime) * sigma_0
                
            else:
                # Simplified for small datasets
                alpha = pm.Normal('alpha', y.mean(), y.std())
                beta = pm.Normal('beta', 0, global_importance)
                mu = alpha + pm.math.sum(beta * X, axis=1)
                sigma = pm.HalfNormal('sigma', y.std())
            
            print("   🎭 Adding robust mixture likelihood...")
            # INNOVATION 5: Mixture likelihood for robustness against outliers
            normal_component = pm.Normal.dist(mu=mu, sigma=sigma)
            robust_component = pm.StudentT.dist(nu=3, mu=mu, sigma=sigma*1.5)
            
            mixture_weights = pm.Dirichlet('mixture_weights', a=[4, 1])  # Favor normal
            likelihood = pm.Mixture('likelihood', w=mixture_weights,
                                  comp_dists=[normal_component, robust_component],
                                  observed=y)
            
            print("   ⚡ Configuring autonomous sampling...")
            # Sample with autonomous-optimized settings
            trace = pm.sample(2000, tune=1500, chains=4, 
                             target_accept=0.95,
                             nuts={'max_treedepth': 12},
                             random_seed=42, idata_kwargs={'log_likelihood': True})
        
        print("✅ Autonomous hybrid model completed!")
        print("   Key innovations:")
        print("   • Learned feature importance hierarchies")
        print("   • Regime-switching dynamics")
        print("   • Non-linear feature transformations")
        print("   • Adaptive error modeling")
        print("   • Robust mixture likelihoods")
        
        return autonomous_model, trace
    
    def save_results(self, model, trace, model_name):
        """Save model results with comprehensive diagnostics"""
        print(f"\n💾 Saving results for {model_name}...")
        
        # Compute diagnostics
        summary = az.summary(trace)
        
        # Save MCMC diagnostics
        diagnostics_path = f'./files/analysis/mcmc_diagnostics_{model_name}.csv'
        os.makedirs(os.path.dirname(diagnostics_path), exist_ok=True)
        summary.to_csv(diagnostics_path)
        
        # Compute model comparison metrics
        waic = az.waic(trace, model)
        loo = az.loo(trace, model)
        
        # Save model comparison results
        comparison_path = f'./files/analysis/model_comparison_{model_name}.csv'
        
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
        
        # Print summary
        print(f"\nModel Summary for {model_name}:")
        print(summary)
        print(f"\nWAIC: {waic_value:.2f} ± {waic_se:.2f}")
        print(f"LOO: {loo_value:.2f} ± {loo_se:.2f}")
        
        # Check convergence
        max_rhat = summary['r_hat'].max()
        min_ess = summary['ess_bulk'].min()
        
        print(f"\nConvergence Diagnostics:")
        print(f"Max R-hat: {max_rhat:.4f} (should be < 1.01)")
        print(f"Min ESS: {min_ess:.0f} (should be > 400)")
        
        converged = max_rhat < 1.01 and min_ess > 400
        print(f"Converged: {'✓' if converged else '✗'}")
        
        return converged

def run_advanced_experiment(data_path, target_col=None, model_name="advanced"):
    """Run advanced Bayesian modeling experiment"""
    print("=" * 80)
    print("🧠 ADVANCED BAYESIAN MODELING EXPERIMENT")
    print("=" * 80)
    
    try:
        # Initialize model library
        library = BayesianModelLibrary(data_path, target_col)
        
        # Auto-select and build model
        model, trace = library.auto_select_and_build()
        
        # Save results
        converged = library.save_results(model, trace, model_name)
        
        if converged:
            print(f"\n🎉 Advanced Bayesian experiment completed successfully!")
            return True
        else:
            print(f"\n⚠️  Model converged with issues - check diagnostics")
            return False
            
    except Exception as e:
        print(f"\n❌ Advanced experiment failed: {e}")
        # Save error to debug file
        os.makedirs('./files/debug', exist_ok=True)
        with open('./files/debug/sampling_error.txt', 'w') as f:
            f.write(f"Error: {e}\n")
        return False

if __name__ == "__main__":
    # Test with hierarchical data
    print("Testing with hierarchical dataset...")
    run_advanced_experiment("data/hierarchical_data.csv", target_col="y", model_name="hierarchical")
    
    print("\n" + "="*50 + "\n")
    
    # Test with count data
    print("Testing with count dataset...")
    run_advanced_experiment("data/count_data.csv", target_col="count", model_name="count")