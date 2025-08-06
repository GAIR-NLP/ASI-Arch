"""
Mock agents framework for testing the pipeline without the actual agents library
"""
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import asyncio

class BaseModel:
    """Mock BaseModel for pydantic compatibility"""
    pass

def function_tool(func: Callable) -> Callable:
    """Mock function_tool decorator"""
    return func

class Agent:
    """Mock Agent class"""
    def __init__(self, name: str, instructions: str, tools: List[Any] = None, 
                 output_type: Any = None, model: str = "gpt-4o", max_turns: int = 10):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.output_type = output_type
        self.model = model
        self.max_turns = max_turns
    
    async def run(self, input_data: Any) -> Any:
        """Mock run method that returns a basic response"""
        # For testing, just return a mock successful response
        if self.output_type:
            if hasattr(self.output_type, '__annotations__'):
                # Create a mock output based on the expected type
                fields = self.output_type.__annotations__
                mock_data = {}
                for field, field_type in fields.items():
                    if field_type == str:
                        mock_data[field] = f"Mock {field} response"
                    elif field_type == bool:
                        mock_data[field] = True
                    else:
                        mock_data[field] = f"Mock {field}"
                
                # Create mock object with the expected attributes
                class MockOutput:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)
                
                return MockOutput(**mock_data)
        
        return "Mock agent response"

# Mock result objects
@dataclass
class MockResult:
    final_output: Any
    usage: Optional[Dict] = None

async def log_agent_run(agent_name: str, agent: Agent, input_data: Any, max_turns: int = None) -> MockResult:
    """Mock log_agent_run function"""
    print(f"[MOCK] Running agent: {agent_name}")
    result = await agent.run(input_data)
    return MockResult(final_output=result)

def set_default_openai_client(client: Any) -> None:
    """Mock set_default_openai_client"""
    pass

def set_default_openai_api(api: str) -> None:
    """Mock set_default_openai_api"""
    pass

def set_tracing_disabled(disabled: bool) -> None:
    """Mock set_tracing_disabled"""
    pass

class exceptions:
    """Mock exceptions module"""
    class MaxTurnsExceeded(Exception):
        pass