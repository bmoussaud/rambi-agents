

import asyncio
import logging
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_core.components.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    SystemMessage,
    UserMessage
)
from dotenv import load_dotenv
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_core.components.tools import FunctionTool
from autogen_agentchat.agents import ToolUseAssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.base import CancellationToken
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors import DockerCommandLineCodeExecutor

# Get configuration settings 
load_dotenv()

# Configure logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.DEBUG)

async def get_weather(city: str) -> str:
    return f"The weather in {city} is 72 degrees and Sunny."


async with DockerCommandLineCodeExecutor(work_dir="coding") as code_executor:  # type: ignore[syntax]
    code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
    code_execution_result = code_executor_agent.on_messages(
        messages=[
            TextMessage(content="Here is some code \n ```python print('Hello world') \n``` ", source="user"),
        ],
        cancellation_token=CancellationToken(),
    )
    print(code_execution_result)


async def main():
 
    
    # Create an OpenAI model client.
    # https://27iigguorarqw-openai.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-08-01-preview
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint="https://27iigguorarqw-openai.openai.azure.com/",
        azure_deployment="gpt4o/chat/completions?api-version=2024-08-01-preview",
        model="gpt-4o-2024-08-06",
        api_version="2024-08-01-preview",
        model_capabilities={
            "vision": False, 
            "audio": False,
            "json_output": True,
            "chat": True,
            "function_calling" : True},
        )
    
    get_weather_tool = FunctionTool(get_weather, description="Get the weather for a city")

    
    tool_use_agent = ToolUseAssistantAgent(
        "tool_use_agent",
        system_message="You are a helpful assistant that solves tasks by only using your tools.",
        model_client=model_client,
        registered_tools=[get_weather_tool],
    )
    # Log the request details
    logging.debug("Sending request to Azure OpenAI Chat Completion API")
    tool_result = await tool_use_agent.on_messages(
        messages=[
            TextMessage(content="What is the weather right now in France?", source="user"),
        ],
        cancellation_token=CancellationToken(),
    )
    print(tool_result)

asyncio.run(main())