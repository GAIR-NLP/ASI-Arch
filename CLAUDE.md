# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASI-Arch is an autonomous multi-agent framework for scientific research in Bayesian model discovery and development. The system autonomously hypothesizes, implements, and validates novel probabilistic models using PyMC through a continuous loop of evolution and evaluation. Originally designed for neural architecture research, it has been adapted for autonomous Bayesian statistical modeling.

## Core Architecture

The system consists of three main components working together:

### 1. Pipeline (`pipeline/`)
The autonomous research loop with specialized Bayesian modeling agents:
- **evolve/**: Creates new Bayesian models via `Planner`, `Code Checker`, and `Deduplication` agents
- **eval/**: Validates models through MCMC sampling via `Bayesian Sampler` and `Debugger` agents  
- **analyse/**: Analyzes results with statistical `Analyzer` agent
- **pipeline.py**: Main orchestrator running the continuous Bayesian experiment loop

### 2. Database (`database/`)
MongoDB-based storage system managing all experimental data:
- **mongodb_database.py**: High-level client for database operations
- **candidate_manager.py**: Maintains top-performing Bayesian model candidates
- **faiss_manager.py**: Vector similarity search for model deduplication
- **evaluate_agent/**: Bayesian model scoring and evaluation

### 3. Cognition Base (`cognition_base/`)
RAG-powered knowledge system providing Bayesian methodology insights:
- **cognition/**: Bayesian statistics and methodology paper corpus (100+ JSON files)
- **rag_service.py**: Vector-based knowledge retrieval for statistical guidance
- **rag_api.py**: Flask API for Bayesian modeling knowledge queries

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n asi-arch python=3.10
conda activate asi-arch

# Install dependencies
pip install -r requirements.txt  # Includes PyMC, ArviZ, and other Bayesian tools
pip install -r database/requirements.txt
pip install -r cognition_base/requirements.txt
```

### Running the System
```bash
# Start database service (in separate terminal)
cd database
docker-compose up -d
./start_api.sh

# Start cognition base service (in separate terminal)  
cd cognition_base
docker-compose up -d
python rag_api.py

# Run main Bayesian research pipeline
cd pipeline
python pipeline.py
```

### Key Configuration
- **pipeline/config.py**: Central configuration for PyMC parameters, MCMC settings, and file paths
- **run_bayesian_experiment.sh**: Bayesian model experiment runner script
- **database/docker-compose.yml**: MongoDB container configuration
- **cognition_base/docker-compose.yml**: OpenSearch container configuration

## Important Implementation Details

### Multi-Agent System
The pipeline implements a sophisticated agent-based architecture where specialized LLM agents handle Bayesian modeling tasks:
- Agents use OpenAI/Azure OpenAI APIs configured in pipeline.py
- Each module (evolve, eval, analyse) contains Bayesian-specific prompt templates and model classes
- Agent communication is asynchronous using Python's asyncio
- Agents specialize in PyMC model generation, MCMC diagnostics, and statistical analysis

### Data Flow
1. **Sample**: Retrieve effective parent Bayesian model from database
2. **Evolve**: Generate novel PyMC model via evolutionary agents
3. **Evaluate**: Run MCMC sampling and convergence diagnostics
4. **Analyze**: Generate statistical insights comparing to baselines  
5. **Update**: Store results and update candidate model set

### Error Handling
- Pipeline includes retry mechanisms (MAX_RETRY_ATTEMPTS: 10)
- Debugger agent can automatically fix MCMC sampling errors (MAX_DEBUG_ATTEMPT: 3)
- Comprehensive logging system tracks all pipeline stages
- Convergence monitoring with configurable thresholds (R-hat < 1.01, ESS > 400)

### External Dependencies
- PyMC 5.x for Bayesian modeling and MCMC sampling
- ArviZ for Bayesian data analysis and diagnostics
- MongoDB for persistent storage
- OpenSearch for vector similarity search
- Docker for containerized services
- FAISS for efficient similarity matching
- Optional: GPU acceleration for large-scale MCMC sampling

## Working with the Code

When modifying this codebase:
- The system is designed for continuous autonomous Bayesian research
- Each component can run independently but requires the others for full functionality
- Configuration changes should be made in pipeline/config.py (includes MCMC parameters)
- New agents should follow the existing pattern with Bayesian-specific prompts
- Database schema is defined through the DataElement class structure (includes MCMC diagnostics fields)
- Model files are stored as PyMC implementations in the MODEL_POOL directory
- Experiment scripts should follow the Bayesian workflow: model → sampling → diagnostics → analysis