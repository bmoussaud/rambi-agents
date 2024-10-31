import asyncio
import logging
from typing import List, Sequence

from autogen_agentchat.agents import (
    CodingAssistantAgent,
    ToolUseAssistantAgent,
)
from autogen_agentchat.logging import ConsoleLogHandler

from autogen_agentchat.messages import ChatMessage, StopMessage, TextMessage
from autogen_agentchat.teams import MaxMessageTermination, StopMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.base import CancellationToken
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import BaseChatAgent

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



class UserProxyAgent2(BaseChatAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name, "A human user.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage, StopMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> ChatMessage:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "Enter your response: ")
        if "TERMINATE" in user_input:
            return StopMessage(chat_message=StopMessage(content="User has terminated the conversation.", source=self.name))
        return TextMessage(chat_message=TextMessage(content=user_input, source=self.name))
    
class UserProxyAgent(BaseChatAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name, "A human user.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage, StopMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> ChatMessage:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "Enter your response: ")
        if "TERMINATE" in user_input:
            return StopMessage(content="User has terminated the conversation.", source=self.name)
        return TextMessage(content=user_input, source=self.name)

async def flight_search(start: str, destination: str, date: str) -> str:
    return "\n".join(
        [
            f"AC24 from {start} to {destination} on {date} is $500",
            f"UA23 from {start} to {destination} on {date} is $450",
            f"AL21 from {start} to {destination} on {date} is $400",
        ]
    )


async def flight_booking(flight: str, date: str) -> str:
    return f"Booked flight {flight} on {date}"

async def main():
    user_proxy = UserProxyAgent("Benoit")
    flight_broker = ToolUseAssistantAgent(
        "FlightBroker",
        description="An assistant for booking flights",
        model_client=model_client,
        registered_tools=[
            FunctionTool(flight_search, description="Search for flights"),
            FunctionTool(flight_booking, description="Book a flight"),
        ],
    )
    travel_assistant = CodingAssistantAgent(
        "TravelAssistant",
        description="A travel assistant",
        model_client=model_client,
        system_message="You are a travel assistant.",
    )
    team = SelectorGroupChat(
        [user_proxy, flight_broker, travel_assistant], 
        model_client=model_client
    )
    await team.run("Help user plan a trip and book a flight.", termination_condition=StopMessageTermination())

asyncio.run(main())