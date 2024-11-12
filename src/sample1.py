

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
import os
from autogen_agentchat.agents import AssistantAgent
# Get configuration settings
load_dotenv()

# Configure logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.DEBUG)


async def get_weather(city: str) -> str:
    return f"The weather in {city} is 72 degrees and Sunny."


async def main():

    # Create an OpenAI model client.
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt4o/chat/completions?api-version=2024-08-01-preview",
        model="gpt-4o-2024-08-06",
        api_version="2024-08-01-preview",
        model_capabilities={
            "vision": False,
            "audio": False,
            "json_output": True,
            "chat": True,
            "function_calling": True},
    )

    agent = AssistantAgent(
        name="assistant", model_client=model_client, tools=[get_weather])

    stream = agent.on_messages_stream(
        [TextMessage(content="What is the weather right now in Paris?", source="user")], CancellationToken())

    async for message in stream:
        print(message)


asyncio.run(main())
