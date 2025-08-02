#!/bin/bash

# Bayesian Model Experiment Runner
# Usage: ./run_bayesian_experiment.sh <model_name>

set -e

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if model name is provided
if [ $# -eq 0 ]; then
    log_error "No model name provided"
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL_NAME=$1
log_info "Running Bayesian experiment for model: $MODEL_NAME"

# Set up directories
ANALYSIS_DIR="./files/analysis"
DEBUG_DIR="./files/debug"
EDA_DIR="./files/eda"
PPC_DIR="./files/analysis/posterior_predictive"

# Create directories if they don't exist
mkdir -p "$ANALYSIS_DIR"
mkdir -p "$DEBUG_DIR"
mkdir -p "$EDA_DIR"
mkdir -p "$PPC_DIR"

# Clear previous debug file
> "$DEBUG_DIR/sampling_error.txt"

# Function to run Python script with error handling
run_python_with_logging() {
    local script_name=$1
    local description=$2
    
    log_info "$description"
    
    if pixi run python "$script_name" 2>>"$DEBUG_DIR/sampling_error.txt"; then
        log_success "$description completed successfully"
        return 0
    else
        log_error "$description failed"
        echo "Error details:" >> "$DEBUG_DIR/sampling_error.txt"
        echo "Script: $script_name" >> "$DEBUG_DIR/sampling_error.txt"
        echo "Timestamp: $(date)" >> "$DEBUG_DIR/sampling_error.txt"
        echo "---" >> "$DEBUG_DIR/sampling_error.txt"
        return 1
    fi
}

# Main experiment execution
main() {
    log_info "Starting Bayesian model experiment pipeline"
    
    # Check if PyMC model file exists
    if [ ! -f "pymc_model.py" ]; then
        log_error "PyMC model file 'pymc_model.py' not found"
        echo "ERROR: PyMC model file not found" > "$DEBUG_DIR/sampling_error.txt"
        exit 1
    fi
    
    # Step 1: Run the PyMC model
    log_info "Step 1: Running PyMC model sampling"
    if ! run_python_with_logging "pymc_model.py" "PyMC model sampling"; then
        log_error "PyMC model sampling failed"
        exit 1
    fi
    
    # Step 2: Check if results files were generated
    log_info "Step 2: Validating experiment outputs"
    
    if [ -f "$ANALYSIS_DIR/mcmc_diagnostics.csv" ]; then
        log_success "MCMC diagnostics file generated"
    else
        log_warning "MCMC diagnostics file not found"
    fi
    
    if [ -f "$ANALYSIS_DIR/model_comparison.csv" ]; then
        log_success "Model comparison file generated"
    else
        log_warning "Model comparison file not found"
    fi
    
    # Step 3: Run convergence diagnostics
    log_info "Step 3: Running convergence diagnostics"
    
    # Create a simple Python script to check convergence
    cat > check_convergence.py << 'EOF'
import pandas as pd
import sys
import os

def check_convergence():
    """Check MCMC convergence diagnostics"""
    try:
        # Try to read diagnostics file
        if os.path.exists('./files/analysis/mcmc_diagnostics.csv'):
            df = pd.read_csv('./files/analysis/mcmc_diagnostics.csv')
            
            # Check for required columns (ArviZ uses ess_bulk instead of ess)
            required_cols = ['r_hat', 'ess_bulk']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Missing diagnostic columns: {missing_cols}")
                return False
            
            # Check convergence criteria
            rhat_threshold = 1.01
            ess_threshold = 400
            
            max_rhat = df['r_hat'].max() if 'r_hat' in df.columns else 0
            min_ess = df['ess_bulk'].min() if 'ess_bulk' in df.columns else 1000
            
            print(f"Max R-hat: {max_rhat:.4f} (threshold: {rhat_threshold})")
            print(f"Min ESS: {min_ess:.0f} (threshold: {ess_threshold})")
            
            converged = max_rhat < rhat_threshold and min_ess > ess_threshold
            
            if converged:
                print("✓ Model converged successfully")
                return True
            else:
                print("✗ Model did not converge")
                return False
        else:
            print("No diagnostics file found")
            return False
            
    except Exception as e:
        print(f"Error checking convergence: {e}")
        return False

if __name__ == "__main__":
    success = check_convergence()
    sys.exit(0 if success else 1)
EOF
    
    if pixi run python check_convergence.py 2>>"$DEBUG_DIR/sampling_error.txt"; then
        log_success "Model converged successfully"
    else
        log_warning "Model convergence check failed - see diagnostics"
    fi
    
    # Step 4: Clean up temporary files
    rm -f check_convergence.py
    
    # Step 5: Final validation
    log_info "Step 4: Final experiment validation"
    
    # Check if we have the minimum required outputs
    required_files=("$ANALYSIS_DIR/mcmc_diagnostics.csv")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        log_success "Bayesian experiment completed successfully for model: $MODEL_NAME"
        log_info "Results saved in: $ANALYSIS_DIR"
        exit 0
    else
        log_error "Experiment incomplete - missing required files:"
        for file in "${missing_files[@]}"; do
            log_error "  - $file"
        done
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files"
    rm -f check_convergence.py
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"