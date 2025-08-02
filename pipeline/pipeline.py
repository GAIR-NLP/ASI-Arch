import asyncio

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncAzureOpenAI

from analyse import analyse
from database import program_sample, update
from eval import evaluation
from evolve import evolve
from utils.agent_logger import end_pipeline, log_error, log_info, log_step, log_warning, start_pipeline

client = AsyncAzureOpenAI()

set_default_openai_client(client)
set_default_openai_api("chat_completions") 

set_tracing_disabled(True)


async def run_single_experiment() -> bool:
    """Run single experiment loop - using pipeline categorized logging."""
    # Start a new pipeline process
    pipeline_id = start_pipeline("experiment")
    
    try:
        # Step 1: Model sampling
        log_step("Model Sampling", "Start sampling model from database")
        context, parent = await program_sample()
        log_info(f"Model sampling completed, context length: {len(str(context))}")
        
        # Step 2: Model Evolution
        log_step("Bayesian Model Evolution", "Start evolving new Bayesian model")
        name, motivation = await evolve(context)
        if name == "Failed":
            log_error("Bayesian model evolution failed")
            end_pipeline(False, "Evolution failed")
            return False
        log_info(f"Bayesian model evolution successful, generated model: {name}")
        log_info(f"Evolution motivation: {motivation}")
        
        # Step 3: Model Evaluation
        log_step("Bayesian Model Evaluation", f"Start evaluating model {name}")
        success = await evaluation(name, motivation)
        if not success:
            log_error(f"Bayesian model {name} evaluation failed")
            end_pipeline(False, "Evaluation failed")
            return False
        log_info(f"Bayesian model {name} evaluation successful")
        
        # Step 4: Statistical Analysis
        log_step("Statistical Result Analysis", f"Start analyzing model {name} results")
        result = await analyse(name, motivation, parent=parent)
        log_info(f"Statistical analysis completed, result: {result}")
        
        # Step 5: Update database
        log_step("Database Update", "Update results to database")
        update(result)
        log_info("Database update completed")
        
        # Successfully complete pipeline
        log_info("Bayesian experiment pipeline completed successfully")
        end_pipeline(True, f"Experiment completed successfully, model: {name}, result: {result}")
        return True
        
    except KeyboardInterrupt:
        log_warning("User interrupted experiment")
        end_pipeline(False, "User interrupted experiment")
        return False
    except Exception as e:
        log_error(f"Experiment pipeline unexpected error: {str(e)}")
        end_pipeline(False, f"Unexpected error: {str(e)}")
        return False


async def main():
    """Main function - continuous experiment execution."""
    set_tracing_disabled(True)
    
    log_info("Starting continuous Bayesian model research pipeline...")
    
    # Run initial setup
    log_info("Running initial setup...")
    log_info("Setup completed")
    
    experiment_count = 0
    while True:
        try:
            experiment_count += 1
            log_info(f"Starting experiment {experiment_count}")
            
            success = await run_single_experiment()
            if success:
                log_info(f"Experiment {experiment_count} completed successfully, starting next experiment...")
            else:
                log_warning(f"Experiment {experiment_count} failed, retrying in 60 seconds...")
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            log_warning("Continuous experiment interrupted by user")
            break
        except Exception as e:
            log_error(f"Main loop unexpected error: {e}")
            log_info("Retrying in 60 seconds...")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())