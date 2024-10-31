import logging
import asyncio

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import CodingAssistantAgent, ToolUseAssistantAgent
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, MaxMessageTermination
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import AzureOpenAIChatCompletionClient

# Get configuration settings 
from dotenv import load_dotenv
load_dotenv()

# Set up a log handler to print logs to the console.$
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.INFO)


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


writing_assistant_agent = CodingAssistantAgent(
    name="writing_assistant_agent",
    system_message="You are a helpful assistant that solve tasks by generating text responses and code.",
    model_client=model_client,
)


async def get_weather(city: str) -> str:
    print (f"----called with {city}!!.\n")
    return f"The weather in {city} is 72 degrees and Sunny."


async def main():

    get_weather_tool = FunctionTool(get_weather, description="Get the weather for a city")

    tool_use_agent = ToolUseAssistantAgent(
        "tool_use_agent",
        system_message="You are a helpful assistant that solves tasks by only using your tools.",
        model_client=model_client,
        registered_tools=[get_weather_tool],
    )


    #round_robin_team = RoundRobinGroupChat([tool_use_agent, writing_assistant_agent])
    #round_robin_team_result = await round_robin_team.run(
    #    "Write a Haiku about the weather in Paris", termination_condition=MaxMessageTermination(max_messages=3)
    #)

    llm_team = SelectorGroupChat([tool_use_agent, writing_assistant_agent], model_client=model_client)

    llm_team_result = await llm_team.run(
        "What is the weather in paris right now? Also write a haiku about it.",
        termination_condition=MaxMessageTermination(max_messages=2),
    )

asyncio.run(main())