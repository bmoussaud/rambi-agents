import asyncio
import logging

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import CodingAssistantAgent
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import MaxMessageTermination, StopMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_ext.models import AzureOpenAIChatCompletionClient

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.INFO)

# Get configuration settings 
from dotenv import load_dotenv
load_dotenv()

async def main():
    # Create an OpenAI model client.
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

    #writing_assistant_agent = CodingAssistantAgent(
    #    name="writing_assistant_agent",
    #    system_message="You are a helpful assistant that solve tasks by generating text responses and code.",
    #    model_client=model_client,
    #)

    writing_assistant_agent = CodingAssistantAgent(
        name="writing_assistant_agent",
        system_message="You are a helpful assistant that solve tasks by generating text responses and code. Respond with TERMINATE when the task is done.",
        model_client=model_client,
    )

    round_robin_team = RoundRobinGroupChat([writing_assistant_agent])
    round_robin_team_result = await round_robin_team.run(
        "Write a unique, Haiku about the weather in Paris", termination_condition=MaxMessageTermination(max_messages=3)
    )

asyncio.run(main())