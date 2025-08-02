class Config:
    """Configuration settings for Bayesian model research experiments."""
    # Target PyMC model file
    SOURCE_FILE: str = "intelligent_pymc_model.py"
    
    # Bayesian model training/sampling script
    BASH_SCRIPT: str = "./run_bayesian_experiment.sh"
    
    # Experiment results
    RESULT_FILE: str = "./files/analysis/mcmc_diagnostics.csv"
    RESULT_FILE_TEST: str = "./files/analysis/model_comparison.csv"
    POSTERIOR_PREDICTIVE_DIR: str = "./files/analysis/posterior_predictive/"
    
    # Debug file
    DEBUG_FILE: str = "./files/debug/sampling_error.txt"
    
    # Model pool directory
    MODEL_POOL: str = "./pool"
    
    # Dataset directory
    DATASET_DIR: str = "./data/"
    
    # EDA results directory
    EDA_DIR: str = "./files/eda/"
    
    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 3
    
    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 10
    
    # MCMC sampling parameters
    MCMC_SAMPLES: int = 2000
    MCMC_TUNE: int = 1000
    MCMC_CHAINS: int = 4
    MCMC_TARGET_ACCEPT: float = 0.8
    
    # Convergence thresholds
    RHAT_THRESHOLD: float = 1.01
    ESS_THRESHOLD: int = 400
    
    # Model comparison settings
    WAIC_THRESHOLD: float = 10.0  # Significant difference threshold
    
    # RAG service URL
    RAG: str = "your rag url"
    
    # Database URL
    DATABASE: str = "your database url"