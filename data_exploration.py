"""
Advanced Data Exploration and Model Selection for Bayesian Modeling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class BayesianModelSelector:
    """Intelligent model selection based on data characteristics"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.target_col = None
        self.feature_cols = None
        self.data_characteristics = {}
        self.model_recommendations = []
        
    def load_and_explore(self, target_col=None):
        """Load data and perform comprehensive EDA"""
        print("🔍 Loading and exploring dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Auto-detect target if not specified (assume last column)
        if target_col is None:
            self.target_col = self.df.columns[-1]
        else:
            self.target_col = target_col
            
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Target variable: {self.target_col}")
        print(f"Features: {self.feature_cols}")
        
        return self._comprehensive_eda()
    
    def _comprehensive_eda(self):
        """Perform comprehensive exploratory data analysis"""
        characteristics = {}
        
        # Basic statistics
        characteristics['n_samples'] = len(self.df)
        characteristics['n_features'] = len(self.feature_cols)
        characteristics['missing_values'] = self.df.isnull().sum().sum()
        characteristics['missing_pct'] = (characteristics['missing_values'] / 
                                        (characteristics['n_samples'] * (characteristics['n_features'] + 1))) * 100
        
        # Target variable analysis
        target = self.df[self.target_col].dropna()
        characteristics['target_type'] = self._detect_variable_type(target)
        characteristics['target_distribution'] = self._analyze_distribution(target)
        
        # Feature analysis
        characteristics['feature_types'] = {}
        characteristics['feature_distributions'] = {}
        characteristics['correlations'] = {}
        
        for col in self.feature_cols:
            if col in self.df.columns:
                feature = self.df[col].dropna()
                characteristics['feature_types'][col] = self._detect_variable_type(feature)
                characteristics['feature_distributions'][col] = self._analyze_distribution(feature)
                
                # Correlation with target (if both numeric)
                if (characteristics['target_type'] in ['continuous', 'discrete'] and 
                    characteristics['feature_types'][col] in ['continuous', 'discrete']):
                    corr = stats.pearsonr(feature, target[self.df[col].notna()])[0]
                    characteristics['correlations'][col] = corr
        
        # Store characteristics before pattern detection
        self.data_characteristics = characteristics
        
        # Detect patterns
        characteristics['patterns'] = self._detect_patterns()
        
        # Complexity assessment
        characteristics['complexity'] = self._assess_complexity()
        
        self.data_characteristics = characteristics
        return characteristics
    
    def _detect_variable_type(self, series):
        """Detect if variable is continuous, discrete, categorical, or binary"""
        unique_vals = series.nunique()
        n_samples = len(series)
        
        if series.dtype == 'object' or series.dtype.name == 'category':
            return 'categorical'
        elif unique_vals == 2:
            return 'binary'
        elif unique_vals <= 0.05 * n_samples and unique_vals <= 20:
            return 'discrete'
        else:
            return 'continuous'
    
    def _analyze_distribution(self, series):
        """Analyze the distribution characteristics of a variable"""
        if series.dtype == 'object':
            return {'type': 'categorical', 'categories': series.value_counts().to_dict()}
        
        # Numerical analysis
        analysis = {
            'mean': series.mean(),
            'std': series.std(),
            'skewness': stats.skew(series),
            'kurtosis': stats.kurtosis(series),
            'range': series.max() - series.min(),
            'iqr': series.quantile(0.75) - series.quantile(0.25)
        }
        
        # Distribution tests
        analysis['normality_p'] = stats.normaltest(series)[1] if len(series) > 8 else None
        analysis['is_normal'] = analysis['normality_p'] > 0.05 if analysis['normality_p'] else False
        
        # Detect potential distribution family
        if analysis['is_normal']:
            analysis['likely_distribution'] = 'normal'
        elif series.min() >= 0 and analysis['skewness'] > 1:
            analysis['likely_distribution'] = 'lognormal'
        elif series.min() >= 0 and analysis['mean'] < analysis['std']**2:
            analysis['likely_distribution'] = 'gamma'
        elif all(series == series.astype(int)) and series.min() >= 0:
            analysis['likely_distribution'] = 'poisson'
        else:
            analysis['likely_distribution'] = 'unknown'
            
        return analysis
    
    def _detect_patterns(self):
        """Detect complex patterns in the data"""
        patterns = {}
        
        # Non-linear relationships
        patterns['nonlinear_relationships'] = []
        for col in self.feature_cols:
            if (col in self.df.columns and 
                self.data_characteristics['feature_types'][col] in ['continuous', 'discrete'] and
                self.data_characteristics['target_type'] in ['continuous', 'discrete']):
                
                x = self.df[col].dropna()
                y = self.df[self.target_col][self.df[col].notna()]
                
                # Polynomial correlation test
                linear_corr = abs(stats.pearsonr(x, y)[0])
                poly_corr = abs(stats.pearsonr(x**2, y)[0])
                
                if poly_corr > linear_corr + 0.1:
                    patterns['nonlinear_relationships'].append({
                        'feature': col,
                        'type': 'polynomial',
                        'strength': poly_corr
                    })
        
        # Hierarchical structure detection
        categorical_cols = [col for col, type_info in self.data_characteristics['feature_types'].items() 
                           if type_info == 'categorical']
        patterns['hierarchical_structure'] = len(categorical_cols) > 1
        patterns['categorical_features'] = categorical_cols
        
        # Temporal patterns (if date-like columns exist)
        date_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                date_cols.append(col)
        patterns['temporal'] = len(date_cols) > 0
        patterns['date_columns'] = date_cols
        
        # Multimodality detection
        if self.data_characteristics['target_type'] == 'continuous':
            target = self.df[self.target_col].dropna()
            # Simple bimodality test using Hartigan's dip test approximation
            hist, _ = np.histogram(target, bins=30)
            peaks = len([i for i in range(1, len(hist)-1) 
                        if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
            patterns['multimodal'] = peaks > 1
            patterns['n_modes'] = peaks
        
        return patterns
    
    def _assess_complexity(self):
        """Assess the complexity of the modeling problem"""
        complexity = {}
        
        # Size complexity
        n_features = self.data_characteristics['n_features']
        n_samples = self.data_characteristics['n_samples']
        
        if n_features < 5:
            complexity['feature_complexity'] = 'low'
        elif n_features < 20:
            complexity['feature_complexity'] = 'medium'
        else:
            complexity['feature_complexity'] = 'high'
            
        # Sample size adequacy
        if n_samples / n_features > 50:
            complexity['sample_adequacy'] = 'excellent'
        elif n_samples / n_features > 20:
            complexity['sample_adequacy'] = 'good'
        elif n_samples / n_features > 5:
            complexity['sample_adequacy'] = 'adequate'
        else:
            complexity['sample_adequacy'] = 'poor'
        
        # Relationship complexity
        nonlinear_count = len(self.data_characteristics['patterns']['nonlinear_relationships'])
        if nonlinear_count == 0:
            complexity['relationship_complexity'] = 'linear'
        elif nonlinear_count <= 2:
            complexity['relationship_complexity'] = 'moderate_nonlinear'
        else:
            complexity['relationship_complexity'] = 'highly_nonlinear'
        
        # Overall complexity score (1-10)
        complexity_score = 1
        if complexity['feature_complexity'] == 'high':
            complexity_score += 3
        elif complexity['feature_complexity'] == 'medium':
            complexity_score += 1
            
        if complexity['relationship_complexity'] == 'highly_nonlinear':
            complexity_score += 3
        elif complexity['relationship_complexity'] == 'moderate_nonlinear':
            complexity_score += 2
            
        if self.data_characteristics['patterns']['hierarchical_structure']:
            complexity_score += 2
            
        if self.data_characteristics['patterns']['multimodal']:
            complexity_score += 1
            
        complexity['overall_score'] = min(complexity_score, 10)
        
        return complexity
    
    def recommend_models(self):
        """Recommend appropriate Bayesian models based on data characteristics"""
        recommendations = []
        
        target_type = self.data_characteristics['target_type']
        complexity = self.data_characteristics['complexity']
        patterns = self.data_characteristics['patterns']
        
        # Basic model selection based on target type
        if target_type == 'continuous':
            if complexity['relationship_complexity'] == 'linear':
                recommendations.append({
                    'model_type': 'linear_regression',
                    'reason': 'Linear relationships with continuous target',
                    'complexity': 'low',
                    'priority': 1
                })
            else:
                recommendations.append({
                    'model_type': 'polynomial_regression',
                    'reason': 'Non-linear relationships detected',
                    'complexity': 'medium',
                    'priority': 2
                })
        
        elif target_type == 'binary':
            recommendations.append({
                'model_type': 'logistic_regression',
                'reason': 'Binary target variable',
                'complexity': 'medium',
                'priority': 1
            })
        
        elif target_type == 'discrete':
            recommendations.append({
                'model_type': 'poisson_regression',
                'reason': 'Count/discrete target variable',
                'complexity': 'medium',
                'priority': 1
            })
        
        # Advanced models based on patterns
        if patterns['hierarchical_structure']:
            recommendations.append({
                'model_type': 'hierarchical_model',
                'reason': f'Hierarchical structure detected with categorical features: {patterns["categorical_features"]}',
                'complexity': 'high',
                'priority': 3
            })
        
        if patterns['multimodal']:
            recommendations.append({
                'model_type': 'mixture_model',
                'reason': f'Multimodal distribution detected ({patterns["n_modes"]} modes)',
                'complexity': 'high',
                'priority': 3
            })
        
        if patterns['temporal']:
            recommendations.append({
                'model_type': 'time_series_model',
                'reason': f'Temporal structure detected in columns: {patterns["date_columns"]}',
                'complexity': 'high',
                'priority': 2
            })
        
        if complexity['feature_complexity'] == 'high':
            recommendations.append({
                'model_type': 'regularized_regression',
                'reason': f'High-dimensional data ({self.data_characteristics["n_features"]} features)',
                'complexity': 'medium',
                'priority': 2
            })
        
        # Gaussian Process for complex non-linear patterns
        if (complexity['relationship_complexity'] == 'highly_nonlinear' and 
            complexity['sample_adequacy'] in ['good', 'excellent']):
            recommendations.append({
                'model_type': 'gaussian_process',
                'reason': 'Complex non-linear relationships with adequate sample size',
                'complexity': 'high',
                'priority': 3
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        self.model_recommendations = recommendations
        
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive EDA and model selection report"""
        print("=" * 60)
        print("🎯 BAYESIAN MODEL SELECTION REPORT")
        print("=" * 60)
        
        # Data Overview
        print(f"\n📊 DATA OVERVIEW")
        print(f"Shape: {self.data_characteristics['n_samples']} samples × {self.data_characteristics['n_features']} features")
        print(f"Missing values: {self.data_characteristics['missing_values']} ({self.data_characteristics['missing_pct']:.1f}%)")
        print(f"Target variable: {self.target_col} ({self.data_characteristics['target_type']})")
        
        # Target Distribution
        target_info = self.data_characteristics['target_distribution']
        if 'mean' in target_info:
            print(f"Target distribution: {target_info['likely_distribution']}")
            print(f"  Mean: {target_info['mean']:.3f}, Std: {target_info['std']:.3f}")
            print(f"  Skewness: {target_info['skewness']:.3f}, Kurtosis: {target_info['kurtosis']:.3f}")
        
        # Complexity Assessment
        complexity = self.data_characteristics['complexity']
        print(f"\n🧩 COMPLEXITY ASSESSMENT")
        print(f"Feature complexity: {complexity['feature_complexity']}")
        print(f"Sample adequacy: {complexity['sample_adequacy']}")
        print(f"Relationship complexity: {complexity['relationship_complexity']}")
        print(f"Overall complexity score: {complexity['overall_score']}/10")
        
        # Patterns
        patterns = self.data_characteristics['patterns']
        print(f"\n🔍 DETECTED PATTERNS")
        print(f"Non-linear relationships: {len(patterns['nonlinear_relationships'])}")
        if patterns['nonlinear_relationships']:
            for rel in patterns['nonlinear_relationships']:
                print(f"  - {rel['feature']}: {rel['type']} (strength: {rel['strength']:.3f})")
        
        print(f"Hierarchical structure: {'Yes' if patterns['hierarchical_structure'] else 'No'}")
        if patterns['hierarchical_structure']:
            print(f"  Categorical features: {patterns['categorical_features']}")
        
        print(f"Temporal patterns: {'Yes' if patterns['temporal'] else 'No'}")
        if patterns['temporal']:
            print(f"  Date columns: {patterns['date_columns']}")
        
        if 'multimodal' in patterns:
            print(f"Multimodal distribution: {'Yes' if patterns['multimodal'] else 'No'}")
            if patterns['multimodal']:
                print(f"  Number of modes: {patterns['n_modes']}")
        
        # Model Recommendations
        print(f"\n🎯 RECOMMENDED BAYESIAN MODELS")
        print(f"Found {len(self.model_recommendations)} suitable models:")
        
        for i, rec in enumerate(self.model_recommendations, 1):
            print(f"\n{i}. {rec['model_type'].upper()} (Priority {rec['priority']})")
            print(f"   Reason: {rec['reason']}")
            print(f"   Complexity: {rec['complexity']}")
        
        return self.model_recommendations

def analyze_dataset(data_path, target_col=None):
    """Main function to analyze dataset and recommend models"""
    selector = BayesianModelSelector(data_path)
    selector.load_and_explore(target_col)
    recommendations = selector.recommend_models()
    selector.generate_report()
    
    return selector, recommendations

if __name__ == "__main__":
    # Test with the simple dataset
    selector, recs = analyze_dataset("data/test_data.csv", target_col="y")