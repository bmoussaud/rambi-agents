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
import json
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
from autogen_agentchat.base import TaskResult
from autogen_agentchat.task import MaxMessageTermination, StopMessageTermination
from autogen_ext.models import AzureOpenAIChatCompletionClient
from dataclasses import dataclass
from promptflow.tracing import start_trace
from autogen_agentchat.teams import Swarm
from tmdbv3api import Search
from openai import AzureOpenAI

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
    title: str
    plot: str
    posterUrl: str


class ImageGeneratorAgent(AssistantAgent):
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient, handoffs=[]) -> None:
        super().__init__(name=name, model_client=model_client, handoffs=handoffs, tools=[
            FunctionTool(self.generate_movie_poster,
                         description="generate a movie poster based on its description"),
        ], description="An agent that can generate a movie poster based.")

    async def generate_movie_poster(self, posterDescription: str) -> str:

        print(
            f"----generate_movie_poster called with {posterDescription}!!.\n")
        try:
            client = AzureOpenAI()
            response = client.images.generate(
                model="dall-e-3",
                prompt=posterDescription,
                n=1,
                size='1024x1792'
            )
            json_response = json.loads(response.model_dump_json())
            url = json_response["data"][0]["url"]
        except Exception as e:
            print(f"--- Generation Image Error: {e}")
            url = "https://bit.ly/3YOrHPI"
        return url


class ImageGeneratorAgentOLD(BaseChatAgent):
    def __init__(self,  name: str) -> None:
        super().__init__(name=name,
                         description="An agent that can generate a movie poster based.")

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> ChatMessage:
        print(f"\n***** DALLE 3 on_messages  {messages} )) .\n")
        print("\nBEGIN on_messages called with messages: \n")
        for message in messages:
            print("\n-- MessageContent: "+message.content)
            print("\n-- MessageSource: "+message.source)
        print("\nEND on_messages called with messages: \n")
        # url = "https://dalleprodsec.blob.core.windows.net/private/images/2516e58b-2b3b-48ab-a9ed-11d812ed5bd4/generated_00.png?se=2024-11-13T16%3A28%3A38Z&sig=91pS4Js9gQyyiFjediYrZTbxeWrbuD9DDPaXn1FUJ0Y%3D&ske=2024-11-17T20%3A55%3A04Z&skoid=e52d5ed7-0657-4f62-bc12-7e5dbb260a96&sks=b&skt=2024-11-10T20%3A55%3A04Z&sktid=33e01921-4d64-4f8c-a055-5bdaffd5e33d&skv=2020-10-02&sp=r&spr=https&sr=b&sv=2020-10-02"
        url = "https://bit.ly/3YOrHPI"
        content = f"""The new poster is ![alt new movie poster]({
            url} "New Movie Poster")"""
        return TextMessage(content=content, source=self.name)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Resets the agent to its initialization state."""
        print("Reset ImageGeneratorAgent.....")


class ImageDescribeAgent(AssistantAgent):
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient, handoffs=[]) -> None:
        super().__init__(name=name, model_client=model_client, handoffs=handoffs, tools=[
            FunctionTool(self.describe_movie_poster,
                         description="Describe a movie poster based on an URL"),
        ], description="An agent that can describe images based on an URL, for example, it can describe movie posters.")

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
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient, handoffs=[]) -> None:
        super().__init__(name=name, model_client=model_client, handoffs=handoffs, tools=[
            FunctionTool(self.get_movie_plot,
                         description="Get the plot of a movie using its title"),
        ], description="An agent that can search information about movies in public movie Databases (IMDB, TMDB) (plot, actors, posters, etc.)")

    async def get_movie_plot(self, title: str) -> Movie:
        print("*** get_movie_plot called with title: "+title)
        search_results = Search().movies(title)
        if search_results:
            print("*** get_movie_plot called with len: "+str(len(search_results)))
            sr = search_results[0]  # Return the first result
            return Movie(sr.title, sr.overview, f"https://image.tmdb.org/t/p/original/{sr.poster_path}")
        else:
            return None


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
        "movie_database_agent",
        model_client)
    image_describe_agent = ImageDescribeAgent(
        "describe_poster_agent",
        model_client)
    poster_generator_agent = ImageGeneratorAgent(
        "poster_generator_agent",
        model_client)

    summary_agent = AssistantAgent(
        "summary_agent",
        model_client=model_client,
        description="A helpful assistant that can summarize the info about the generate movie and terminate the conversation")

    termination = TextMentionTermination("TERMINATE")

    agents = [movie_database_agent, image_describe_agent,
              poster_generator_agent, summary_agent]
    team_rr = RoundRobinGroupChat(agents,  termination_condition=termination)
    team = SelectorGroupChat(
        agents, model_client=model_client, termination_condition=termination)
    task = """
    The 2 movies are 'The Fall Guy' and 'Avatar'. 
    Search information about these two movies and display the title, the plot, the posterUrl and a poster description. 
    Based on these 2 movies, generate a new movie with a title, a plot and a poster description.
    """
    stream = team.run_stream(task=task)

    int_count = 0

    async for message in stream:
        # delete file
        if os.path.exists(f"output-{int_count}.md"):
            os.remove(f"output-{int_count}.md")
        with open(f"output-{int_count}.md", "a") as f:
            # if message is an instance of autogen_agentchat.base.TaskResult
            if isinstance(message, TaskResult):
                count = 0
                for msg in message.messages:
                    # if msg is an instance of autogen_agentchat.messages.TextMessage
                    if isinstance(msg, TextMessage):
                        f.write(f"## Part {count}\n")
                        f.write(f"{msg.content}\n")
                    count = count + 1
            else:
                f.write(f"{message}\n")
        print(f"## {int_count} ----\n")
        print(message)
        print(f"/## {int_count} ----\n")
        int_count += 1

    print("== DONE")


asyncio.run(main())
