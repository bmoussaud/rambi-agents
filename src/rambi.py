from autogen_agentchat.agents import CodingAssistantAgent
from autogen_agentchat.teams import MaxMessageTermination, StopMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_core.components.tools import FunctionTool
from autogen_agentchat.agents import ToolUseAssistantAgent
from dotenv import load_dotenv
from autogen_agentchat.teams import SelectorGroupChat
import logging
import asyncio
from typing import List, Sequence
from autogen_agentchat.messages import ChatMessage, StopMessage, TextMessage
from autogen_agentchat.teams import MaxMessageTermination, StopMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.base import CancellationToken
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import OpenAIChatCompletionClient
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import BaseChatAgent
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import MaxMessageTermination
from autogen_ext.models import AzureOpenAIChatCompletionClient

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.DEBUG)

# Get configuration settings
load_dotenv()

# Get configuration settings
load_dotenv()

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

async def get_movie_plot(title: str) -> str:
    print (f"\n----get_movie_plot called with {title}!!.\n")
    if title == "Inception":
        return "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."
    elif title == "Avatar":
        return "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."
    elif title == "The Blues Brothers":
        return "Jake Blues, just released from prison, puts together his old band to save the Catholic home where he and his brother Elwood were raised." 
    elif title == "Bambi":
        return "The story of a young deer growing up in the forest."
    return f"I'm sorry, I don't know that the {title} movie."

async def main():
    # Create an OpenAI model client.
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint="https://27iigguorarqw-openai.openai.azure.com/",
        azure_deployment="gpt4o/chat/completions?api-version=2024-08-01-preview",
        model="gpt-4o-2024-08-06",
        api_version="2024-08-01-preview",
        model_capabilities={
            "vision": True,
            "audio": False,
            "json_output": True,
            "chat": True,
            "function_calling": True},
        )

    movie_database = ToolUseAssistantAgent(
        "movie_database",
        model_client=model_client,
        description="An agent that can search information about movies (plot, actors, etc.)",
        system_message="You are a helpful AI assistant. Solve tasks using your tools.",
        registered_tools=[
            FunctionTool(get_movie_plot, description="Get the plot of a movie"),
        ],
    )

    movie_advisor = CodingAssistantAgent(
        "movie_advisor",
        model_client=model_client,
        description="Assist in creating a new movie plot based on 2 existing movies",
        system_message="You are a helpful assistant that can suggest a new movie plot for a user based on 2 existing movies but have no knowledge about existing movies.",
    )

    french_translation_agent = CodingAssistantAgent(
        "french_translation_agent",
        model_client=model_client,
        description="A helpful assistant that translates text into French",
        system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
    )

    summary_agent = CodingAssistantAgent(
        "summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the new movie.",
        system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed tfinal travel plan. You must ensure th b at the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
    )

    group_chat = SelectorGroupChat([UserProxyAgent("Benoit"), movie_database, movie_advisor], model_client=model_client)
    #group_chat = RoundRobinGroupChat([movie_database, movie_advisor, french_translation_agent, summary_agent])
    result = await group_chat.run(task="Imagine a new movie based on 2 existing movies. Result in French", termination_condition=StopMessageTermination())
    #print(result)
    # Access the messages attribute directly
    for message in result.messages:
        print(message.content)
        print("-----\n")


asyncio.run(main())