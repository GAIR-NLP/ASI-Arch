import os
from typing import Tuple

from config import Config
from utils.agent_logger import log_agent_run
from .model import debugger, trainer
from .prompts import Debugger_input


async def evaluation(name: str, motivation: str) -> bool:
    """
    Evaluate Bayesian model performance for a given experiment.
    
    Args:
        name: Experiment name
        motivation: Experiment motivation
        
    Returns:
        True if model sampling and evaluation successful, False otherwise
    """
    success, error_msg = await run_bayesian_sampling(name, motivation)
    if not success:
        print(f"Bayesian model sampling failed: {error_msg}")
        return False
    save_model(name)
    return True


async def run_bayesian_sampling(name: str, motivation: str) -> Tuple[bool, str]:
    """
    Run Bayesian model sampling with debugging retry mechanism.
    
    Args:
        name: Experiment name
        motivation: Experiment motivation
        
    Returns:
        Tuple of (success_flag, error_message)
    """
    try:
        debug = False
        previous_error = ""
        
        for attempt in range(Config.MAX_DEBUG_ATTEMPT):
            if debug:
                debug_result = await log_agent_run(
                    "debugger",
                    debugger,
                    Debugger_input(motivation, previous_error)
                )
                
                changes_made = debug_result.final_output.changes_made
                print(f"Debug changes for {name}: {changes_made}")

            sampling_result = await log_agent_run(
                "bayesian_sampler",
                trainer,  # Reusing trainer agent but with new instructions
                f"""Please run the Bayesian model sampling:
                1. Execute bash {Config.BASH_SCRIPT} with parameter: {name}
                2. Check for MCMC convergence (R-hat < {Config.RHAT_THRESHOLD}, ESS > {Config.ESS_THRESHOLD})
                3. Only return success=True if sampling converges and diagnostics pass"""
            )
            
            if sampling_result.final_output.success:
                print(f"Bayesian model sampling successful for {name}")
                return True, ""
            else:
                debug = True
                # Read debug file content as detailed error information
                try:
                    # If debug file doesn't exist, create an empty file
                    if not os.path.exists(Config.DEBUG_FILE):
                        with open(Config.DEBUG_FILE, 'w', encoding='utf-8') as f:
                            f.write("")

                    with open(Config.DEBUG_FILE, 'r', encoding='utf-8') as f:
                        debug_content = f.read()
                    previous_error = f"Sampling failed. Debug info:\n{debug_content}"
                except Exception as e:
                    previous_error = (
                        f"Sampling failed. Cannot read debug file {Config.DEBUG_FILE}: {str(e)}"
                    )
                
                print(f"Sampling failed for {name} (attempt {attempt + 1}): {previous_error}")
                
                # If this is the last attempt, return failure
                if attempt == Config.MAX_DEBUG_ATTEMPT - 1:
                    return False, (
                        f"Sampling failed after {Config.MAX_DEBUG_ATTEMPT} attempts. "
                        f"Final error: {previous_error}"
                    )
                
                continue
                
    except Exception as e:
        error_msg = f"Unexpected error during sampling: {str(e)}"
        print(error_msg)
        return False, error_msg


def save_model(name: str) -> None:
    """
    Save PyMC model file content to model pool with given name.
    
    Args:
        name: File name to save as
    """
    with open(Config.SOURCE_FILE, "r", encoding='utf-8') as f:
        content = f.read()
    with open(f"{Config.MODEL_POOL}/{name}.py", "w", encoding='utf-8') as f:
        f.write(content)