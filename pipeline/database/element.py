from dataclasses import dataclass, asdict
from typing import Dict, Optional

from utils.agent_logger import log_agent_run
from .model import summarizer
from .prompt import Summary_input


@dataclass
class DataElement:
    """Data element model for Bayesian model experimental results."""
    time: str
    name: str
    result: Dict[str, str]  # Contains convergence_diagnostics, model_comparison_metrics, posterior_summary
    program: str  # PyMC model code
    motivation: str
    analysis: str
    cognition: str
    log: str
    parent: Optional[int] = None
    index: Optional[int] = None
    summary: Optional[str] = None
    # Bayesian-specific fields
    mcmc_diagnostics: Optional[Dict[str, float]] = None  # r_hat, ess, etc.
    model_comparison: Optional[Dict[str, float]] = None  # WAIC, LOO, marginal likelihood
    posterior_predictive: Optional[Dict[str, str]] = None  # PPC results and plots
    dataset_info: Optional[Dict[str, str]] = None  # Dataset metadata and EDA results
    
    def to_dict(self) -> Dict:
        """Convert DataElement instance to dictionary."""
        return asdict(self)
    
    async def get_context(self) -> str:
        """Generate enhanced context with structured experimental evidence presentation."""
        summary = await log_agent_run(
            "summarizer",
            summarizer,
            Summary_input(self.motivation, self.analysis, self.cognition)
        )
        summary_result = summary.final_output.experience

        # Build Bayesian-specific context
        diagnostics_str = ""
        if self.mcmc_diagnostics:
            diagnostics_str = f"""
#### MCMC Convergence Diagnostics
**R-hat**: {self.mcmc_diagnostics.get('r_hat', 'N/A')}
**Effective Sample Size**: {self.mcmc_diagnostics.get('ess', 'N/A')}
**MCMC Divergences**: {self.mcmc_diagnostics.get('divergences', 'N/A')}
"""

        comparison_str = ""
        if self.model_comparison:
            comparison_str = f"""
#### Model Comparison Metrics
**WAIC**: {self.model_comparison.get('waic', 'N/A')}
**LOO**: {self.model_comparison.get('loo', 'N/A')}
**Marginal Likelihood**: {self.model_comparison.get('marginal_likelihood', 'N/A')}
"""

        dataset_str = ""
        if self.dataset_info:
            dataset_str = f"""
#### Dataset Information
**Shape**: {self.dataset_info.get('shape', 'N/A')}
**Missing Values**: {self.dataset_info.get('missing_values', 'N/A')}
**Target Distribution**: {self.dataset_info.get('target_distribution', 'N/A')}
"""

        return f"""## BAYESIAN MODEL EXPERIMENTAL EVIDENCE

### Experiment: {self.name}
**Model Identifier**: {self.name}
{dataset_str}
#### Performance Metrics Summary
**Model Fit**: {self.result.get("model_fit", "N/A")}
**Predictive Performance**: {self.result.get("predictive_performance", "N/A")}
{diagnostics_str}{comparison_str}
#### PyMC Model Implementation
```python
{self.program}
```

#### Synthesized Experimental Insights
{summary_result}

---"""

    @classmethod
    def from_dict(cls, data: Dict) -> 'DataElement':
        """Create DataElement instance from dictionary."""
        return cls(**data)