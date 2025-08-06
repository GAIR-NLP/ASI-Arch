import subprocess
import pandas as pd
import numpy as np
from typing import Any, Dict

from agents import function_tool
from config import Config


@function_tool
def read_pymc_model() -> Dict[str, Any]:
    """Read a PyMC model file and return its contents."""
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def read_csv_file(file_path: str) -> Dict[str, Any]:
    """Read a CSV file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {
            'success': True,
            'content': content
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def write_pymc_model(content: str) -> Dict[str, Any]:
    """Write PyMC model content to file."""
    source_file = Config.SOURCE_FILE
    try:
        with open(source_file, 'w') as f:
            f.write(content)
        return {
            'success': True,
            'message': f'Successfully wrote PyMC model to {source_file}'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@function_tool
def run_bayesian_experiment(name: str, script_path: str) -> Dict[str, Any]:
    """Run the Bayesian model sampling experiment and return its output."""
    try:
        result = subprocess.run(['bash', script_path, name], 
                              capture_output=True, 
                              text=True,
                              check=True)
        return {
            'success': True,
            'output': result.stdout,
            'message': 'Bayesian experiment executed successfully'
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': e.stderr,
            'output': e.stdout
        }


@function_tool
def run_eda_script(dataset_path: str) -> Dict[str, Any]:
    """Run exploratory data analysis on a dataset."""
    try:
        # Basic EDA implementation
        df = pd.read_csv(dataset_path)
        
        eda_results = {
            'shape': str(df.shape),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {col: df[col].value_counts().head().to_dict() 
                                  for col in df.select_dtypes(include=['object']).columns}
        }
        
        return {
            'success': True,
            'eda_results': eda_results
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_rag(query: str) -> Dict[str, Any]:
    """Run RAG and return the results."""
    try:
        import requests
        
        response = requests.post(
            f'{Config.RAG}/search',
            headers={'Content-Type': 'application/json'},
            json={
                'query': query,
                'k': 3, 
                'similarity_threshold': 0.5
            }
        )
        
        response.raise_for_status()
        results = response.json()
        
        return {
            'success': True,
            'results': results
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@function_tool
def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """Load a dataset for Bayesian modeling."""
    try:
        dataset_path = f"{Config.DATASET_DIR}{dataset_name}"
        df = pd.read_csv(dataset_path)
        
        return {
            'success': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'head': df.head().to_dict(),
            'path': dataset_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@function_tool
def validate_pymc_model(model_content: str) -> Dict[str, Any]:
    """Validate PyMC model syntax and structure."""
    try:
        # Basic syntax validation
        import ast
        ast.parse(model_content)
        
        # Check for PyMC imports and basic structure
        required_imports = ['pymc', 'pm']
        has_pymc = any(imp in model_content.lower() for imp in required_imports)
        has_model_context = 'with pm.Model()' in model_content or 'with pymc.Model()' in model_content
        has_sampling = 'pm.sample(' in model_content or 'pymc.sample(' in model_content
        
        validation_results = {
            'syntax_valid': True,
            'has_pymc_import': has_pymc,
            'has_model_context': has_model_context,
            'has_sampling_call': has_sampling
        }
        
        return {
            'success': True,
            'validation': validation_results
        }
    except SyntaxError as e:
        return {
            'success': False,
            'syntax_error': str(e),
            'validation': {'syntax_valid': False}
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }