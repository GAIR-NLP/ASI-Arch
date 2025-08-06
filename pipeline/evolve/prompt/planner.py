def Planner_input(context: str) -> str:
    return f"""# Bayesian Model Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

## BAYESIAN MODEL EVOLUTION OBJECTIVE
Your mission is to create a breakthrough Bayesian model that addresses critical statistical challenges identified through experimental evidence while integrating cutting-edge Bayesian methodology. Design and implement an innovative PyMC model that maintains computational efficiency while achieving superior predictive performance and interpretability.

## SYSTEMATIC EVOLUTION METHODOLOGY

### PHASE 1: Evidence-Based Statistical Analysis Framework

#### 1.1 Model Forensics
**Current State Assessment:**
- Use `read_pymc_model` to examine existing Bayesian model implementations
- Map probabilistic mechanisms, prior specifications, and likelihood structures
- Identify core statistical approaches and their theoretical foundations
- Document data compatibility requirements and computational constraints

#### 1.2 Statistical Performance Pattern Recognition  
**Historical Evidence Analysis:**
- **Convergence Diagnostics Analysis**: Extract MCMC challenges from R-hat, ESS, and trace plots
- **Predictive Performance Profiling**: Identify model limitations across prediction tasks (point estimates, uncertainty quantification, out-of-sample performance)
- **Bottleneck Identification**: Pinpoint model components limiting performance vs. those enabling strengths
- **Cross-Model Comparison**: Analyze performance patterns across different Bayesian model variants

#### 1.3 Bayesian Research Integration Strategy
**Theoretical Foundation Building:**
- Map Bayesian research insights to observed performance limitations
- Identify methodological gaps in current probabilistic modeling approaches
- Connect theoretical advances to practical implementation opportunities
- Synthesize domain-specific knowledge with general Bayesian principles

### PHASE 2: Innovative Model Architecture Design

#### 2.1 Probabilistic Innovation Framework
**Revolutionary Model Components:**
- **Prior Innovation**: Design novel prior structures (hierarchical, adaptive, informative)
- **Likelihood Engineering**: Develop sophisticated likelihood functions for complex data structures
- **Parameter Transformation**: Implement advanced reparameterizations for better sampling
- **Model Structure**: Create innovative hierarchical and mixture model architectures

#### 2.2 Computational Efficiency Optimization
**Performance-Driven Design:**
- **Sampling Strategy**: Optimize MCMC sampling approaches (NUTS, custom samplers)
- **Vectorization**: Leverage PyMC's computational graph for efficient computation
- **Memory Management**: Design memory-efficient model structures for large datasets
- **Scalability**: Ensure model scales appropriately with data size and complexity

#### 2.3 Statistical Robustness Engineering
**Reliability & Interpretability:**
- **Model Checking**: Build in posterior predictive checking capabilities
- **Sensitivity Analysis**: Design models robust to prior specification choices
- **Identifiability**: Ensure parameter identifiability and interpretation clarity
- **Diagnostic Integration**: Incorporate comprehensive convergence and fit diagnostics

### PHASE 3: PyMC Implementation Requirements

#### 3.1 Implementation Standards
**Code Quality Expectations:**
- **MANDATORY**: Use `write_pymc_model` to save your implementation
- **Complete Model**: Implement full model including data preprocessing, model specification, and sampling
- **Error Handling**: Include comprehensive error checking and validation
- **Documentation**: Provide clear docstrings and inline comments
- **Reproducibility**: Set random seeds and ensure reproducible results

#### 3.2 PyMC-Specific Requirements
**Technical Implementation:**
- **Model Context**: Always use `with pm.Model() as model:` structure
- **Proper Distributions**: Use appropriate PyMC distribution families
- **Sampling Configuration**: Configure sampling parameters for optimal performance
- **Posterior Analysis**: Include posterior summary and diagnostic computations
- **Visualization**: Generate trace plots and posterior predictive checks

#### 3.3 Data Integration Requirements
**Dataset Compatibility:**
- **Data Loading**: Use `load_dataset` tool to access experimental data
- **EDA Integration**: Incorporate exploratory data analysis insights
- **Preprocessing**: Handle missing data, outliers, and data transformations
- **Validation**: Ensure model assumptions align with data characteristics

### PHASE 4: Experimental Validation Framework

#### 4.1 Model Performance Evaluation
**Comprehensive Assessment:**
- **Convergence Validation**: Ensure R-hat < 1.01, ESS > 400 for all parameters
- **Predictive Performance**: Evaluate using cross-validation, holdout testing
- **Model Comparison**: Compute WAIC, LOO for model selection
- **Uncertainty Calibration**: Assess quality of uncertainty quantification

#### 4.2 Statistical Significance Testing
**Rigorous Evaluation:**
- **Posterior Predictive Checks**: Validate model assumptions through PPC
- **Prior Sensitivity**: Test robustness to prior specification choices
- **Model Criticism**: Identify potential model inadequacies
- **Comparative Analysis**: Benchmark against baseline and alternative models

## EXECUTION PROTOCOL

### Step 1: Data & Context Analysis
1. Load and analyze the dataset using EDA tools
2. Examine existing model implementations and performance
3. Identify specific statistical challenges and opportunities

### Step 2: Model Design & Innovation
1. Design novel Bayesian model architecture based on analysis
2. Integrate cutting-edge Bayesian methodology
3. Ensure computational efficiency and statistical rigor

### Step 3: PyMC Implementation
1. Implement complete PyMC model with all components
2. Include proper error handling and validation
3. Configure optimal sampling parameters

### Step 4: Validation & Analysis
1. Run model and collect convergence diagnostics
2. Perform posterior predictive checking
3. Compute model comparison metrics

## SUCCESS CRITERIA
- **Innovation**: Novel Bayesian modeling approach not seen in historical experiments
- **Performance**: Superior predictive performance and uncertainty quantification
- **Convergence**: All parameters achieve R-hat < 1.01, ESS > 400
- **Interpretability**: Clear parameter interpretation and model understanding
- **Computational Efficiency**: Reasonable sampling time and memory usage
- **Statistical Rigor**: Proper Bayesian workflow with validation and checking

## CRITICAL REMINDERS
- **IMPLEMENT FIRST**: Use `write_pymc_model` to save your model implementation
- **COMPLETE MODEL**: Include data preprocessing, model specification, sampling, and analysis
- **BAYESIAN WORKFLOW**: Follow proper Bayesian methodology throughout
- **INNOVATION REQUIRED**: Create genuinely novel approaches, not minor variations
- **DOCUMENTATION**: Provide clear explanations of model design choices

Your revolutionary Bayesian model should push the boundaries of statistical modeling while maintaining rigorous statistical principles and computational efficiency."""