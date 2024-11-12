from datetime import datetime
import sys
from autogen_agentchat.agents import CodingAssistantAgent

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_core.components.tools import FunctionTool
from autogen_agentchat.agents import ToolUseAssistantAgent
from dotenv import load_dotenv
from autogen_agentchat.teams import SelectorGroupChat
import logging
import asyncio
import os
from typing import List, Sequence
from autogen_agentchat.messages import ChatMessage, StopMessage, TextMessage, MultiModalMessage
from autogen_agentchat.task import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.base import CancellationToken
from autogen_core.components.tools import FunctionTool
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.task import MaxMessageTermination, StopMessageTermination
from autogen_ext.models import AzureOpenAIChatCompletionClient
from dataclasses import dataclass
from promptflow.tracing import start_trace


logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.DEBUG)

# Get configuration settings
load_dotenv()


# Get the current date and time
current_datetime = datetime.now()

# Format the date and time as a string
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# promptflow tracing
start_trace(collection="autogen-groupchat-"+formatted_datetime)


@dataclass
class Movie:
    plot: str
    posterUrl: str


class ImageGeneratorAgent(BaseChatAgent):
    def __init__(self,  name: str) -> None:
        super().__init__(name=name,
                         description="An agent that can generate images for example a movie poster.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [MultiModalMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> ChatMessage:
        print(f"\n***** DALLE 3 on_messages  {messages} )) .\n")
        print("\nBEGIN on_messages called with messages: \n")
        for message in messages:
            print("\n-- MessageContent: "+message.content)
            print("\n-- MessageSource: "+message.source)
        print("\nEND on_messages called with messages: \n")
        url = "https://dalleprodsec.blob.core.windows.net/private/images/fa2cf45a-5978-4d6d-9717-22439278542b/generated_00.png?se=2024-11-06T13%3A52%3A22Z&sig=WG6RzlkKW5dO%2BfdgXgCEcFy2SOyCmgIHKDR0FmeJSQk%3D&ske=2024-11-11T23%3A42%3A52Z&skoid=e52d5ed7-0657-4f62-bc12-7e5dbb260a96&sks=b&skt=2024-11-04T23%3A42%3A52Z&sktid=33e01921-4d64-4f8c-a055-5bdaffd5e33d&skv=2020-10-02&sp=r&spr=https&sr=b&sv=2020-10-02"

        return MultiModalMessage(source=self.name, content=[url])

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Resets the agent to its initialization state."""
        print("Reset ImageGeneratorAgent.....")


class ImageDescribeAgent(AssistantAgent):
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient) -> None:
        super().__init__(name=name, model_client=model_client, tools=[
            FunctionTool(self.describe_movie_poster,
                         description="Describe a movie poster based on the URL"),
        ], description="An agent that can describe images based on the URL, for example, it can describe movie posters.")

    async def describe_movie_poster(self, posterUrl: str) -> str:
        print(
            f"\n----GPT4O describe_movie_poster called with {posterUrl}!!.\n")
        response = await self._model_client._client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Describe this picture:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": posterUrl
                        }
                    }
                ]}
            ],
            max_tokens=2000
        )
        # Return the generated description
        print(f"\n---GPT4O answer: {response.choices[0].message.content}.\n")
        return response.choices[0].message.content


class MovieDatabaseAgent(AssistantAgent):
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient) -> None:
        super().__init__(name=name, model_client=model_client, tools=[
            FunctionTool(self.get_movie_plot,
                         description="Get the plot of a movie using its title"),
        ], description="An agent that can search information about movies in public movie Databases (IMDB, TMDB) (plot, actors, posters, etc.)")

    # async on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> ChatMessage:

    async def get_movie_plot(self, title: str) -> Movie:
        print(f"\n----SELF get_movie_plot called with {title}!!.\n")
        if title == "Inception":
            plot = "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."
            posterURL = "https://image.tmdb.org/t/p/original/ljsZTbVsrQSqZgWeep2B1QiDKuh.jpg"
        elif title == "Avatar":
            plot = "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home."
            posterURL = "https://image.tmdb.org/t/p/original/kyeqWdyUXW608qlYkRqosgbbJyK.jpg"
        elif title == "The Blues Brothers":
            plot = "Jake Blues, just released from prison, puts together his old band to save the Catholic home where he and his brother Elwood were raised."
            posterURL = "https://image.tmdb.org/t/p/original/rhYJKOt6UrQq7JQgLyQcSWW5R86.jpg"
        elif title == "Bambi":
            plot = "The story of a young deer growing up in the forest."
            posterURL = "https://image.tmdb.org/t/p/original/wV9e2y4myJ4KMFsyFfWYcUOawyK.jpg"
        else:
            plot = f"I'm sorry, I don't know that the {title} movie."
            posterURL = "xxxxx"

        return Movie(plot, posterURL)


async def main():
    # Create an OpenAI model client.
    model_client = AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
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

    movie_database_agent = MovieDatabaseAgent(
        "movie_database_agent", model_client)
    image_describe_agent = ImageDescribeAgent(
        "describe_poster_agent", model_client)
    poster_generator_agent = ImageGeneratorAgent("poster_generator_agent")

    summary_agent = AssistantAgent(
        "summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the new movie.",
        system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide the final answer. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
    )

    termination = TextMentionTermination("TERMINATE")

    team_rr = RoundRobinGroupChat(
        [movie_database_agent, image_describe_agent, summary_agent],  termination_condition=termination)
    team = SelectorGroupChat([movie_database_agent, image_describe_agent, poster_generator_agent, summary_agent],
                             model_client=model_client, termination_condition=termination)
    task = """
    The 2 movies are Bambi and Avatar. 
    Search information about these two movies and display the title, the plot, the posterUrl and a poster description. 
    Based on these 2 movies, generate a new movie with a title, a plot and a poster description.
    """
    stream = team.run_stream(task=task)

    int_count = 0
    async for message in stream:
        print(f"## {int_count} ----\n")
        print(message)
        print(f"/## {int_count} ----\n")
        int_count += 1

    print("== DONE")


asyncio.run(main())
