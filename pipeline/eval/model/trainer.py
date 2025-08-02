from agents import Agent
from pydantic import BaseModel
from tools import run_bayesian_experiment

class SamplingResultOutput(BaseModel):
    success: bool
    error: str

trainer = Agent(
    name="Bayesian Sampler",
    instructions="""You are an expert in running Bayesian model sampling experiments using PyMC.
    Your task is to:
    1. Run the Bayesian experiment script using the provided script and name parameter
    2. Evaluate MCMC convergence diagnostics (R-hat, ESS, divergences)
    3. If sampling is successful and converges properly:
       - Set success=True and leave error empty
       - Convergence criteria: R-hat < 1.01, ESS > 400, minimal divergences
    4. If sampling fails or doesn't converge:
       - Set success=False
       - Analyze the error output and provide a clear explanation of the issue
       - Common issues: poor priors, model misspecification, sampling problems
       
    Focus on identifying statistical and computational issues specific to Bayesian modeling.
    Your error explanation should help with model debugging and MCMC tuning.""",
    tools=[run_bayesian_experiment],
    output_type=SamplingResultOutput,
    model="gpt-4o"
)