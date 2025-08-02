def Analyzer_input(name: str, result: str, motivation: str, ref_context: str) -> str:
    """
    Creates a comprehensive prompt for Bayesian model results analysis with emphasis
    on statistical rigor and critical assessment of probabilistic model innovations.
    """
    return f"""# Bayesian Model Analysis Request: Model {name}

## Resources:
- MCMC Diagnostics & Results: `{result}`
- PyMC model implementation: Use read_pymc_model tool to examine the model
- Statistical motivation: {motivation}

## Related Experiments for Comparative Analysis:
{ref_context}

**IMPORTANT:** The above related experiments represent either parent models (previous iterations that led to this design) or sibling models (alternative approaches explored from the same parent). Use these for comparative analysis to understand:
- What specific probabilistic changes differentiate the current model from its relatives
- Which statistical components are responsible for performance differences
- Whether the modifications represent genuine statistical improvements or trade-offs
- How convergence and fit characteristics compare across related models

## Analysis Requirements:

Please read the results, examine the PyMC implementation using read_pymc_model tool, and analyze the statistical motivation. Your analysis must include:

1. **MODEL DESIGN EVALUATION**
   - Assess the theoretical soundness of the proposed Bayesian model architecture
   - Evaluate whether the PyMC implementation correctly reflects the statistical design intention
   - Identify any gaps between theoretical motivation and actual probabilistic implementation
   - Judge the plausibility of expected statistical improvements based on the model structure
   - Consider prior specifications, likelihood choices, and model hierarchy appropriateness

2. **STATISTICAL RESULTS ANALYSIS**
   - Examine MCMC convergence diagnostics (R-hat, ESS, divergences, energy statistics)
   - Analyze model comparison metrics (WAIC, LOO, marginal likelihood if available)
   - Assess predictive performance through appropriate statistical measures
   - Evaluate posterior predictive check results for model adequacy
   - Compare performance with baseline models and related experiments
   - Identify which aspects of the model succeeded and which failed statistically

3. **EXPECTATION VS REALITY COMPARISON**
   - Compare the theoretical predictions from the statistical motivation with actual results
   - Identify surprising outcomes in convergence, parameter estimates, or predictive performance
   - Analyze whether the Bayesian design hypotheses were supported by the empirical evidence
   - Assess if the probabilistic innovations achieved their intended statistical benefits
   - Highlight discrepancies between expected and observed model behavior

4. **THEORETICAL EXPLANATION WITH EVIDENCE**
   - Provide mechanistic explanations for observed statistical results, supported by:
     * Specific elements of the PyMC code that contributed to performance patterns
     * Statistical theory connecting model design choices to empirical outcomes  
     * Information-theoretic or computational arguments about model efficiency
   - Explain the precise mechanisms behind convergence successes or failures
   - Connect Bayesian methodology principles with the observed diagnostic results
   - Analyze why certain parameters or model components performed better or worse than others

5. **SYNTHESIS AND INSIGHTS**
   - Synthesize key lessons learned about this particular Bayesian modeling approach
   - Identify fundamental trade-offs revealed between model complexity, interpretability, and performance
   - Provide actionable insights for future Bayesian model development and innovation
   - Suggest specific improvements for addressing any convergence or fit limitations identified
   - Discuss broader implications for probabilistic modeling methodology in this domain

## Analysis Standards:
- Ground ALL statistical claims in specific evidence from the diagnostic results and model comparison metrics
- Maintain intellectual honesty about convergence failures, poor model fit, or unexpected results
- Focus on understanding WHY the statistical results occurred, not just describing WHAT happened
- Use proper Bayesian statistical terminology and avoid frequentist misinterpretations
- Provide actionable, evidence-based recommendations for model improvement
- Consider both computational efficiency and statistical validity in your assessment
- Acknowledge limitations and uncertainties in your analysis where appropriate

## Critical Thinking Guidelines:
- Question whether observed improvements are statistically meaningful or due to random variation
- Consider alternative explanations for unexpected results
- Evaluate the generalizability of findings beyond this specific dataset
- Assess the practical significance of statistical improvements
- Be skeptical of overly complex models that may not provide meaningful benefits

Your analysis should demonstrate deep understanding of both Bayesian statistical principles and practical probabilistic modeling challenges, providing insights that will advance the field of computational statistics."""