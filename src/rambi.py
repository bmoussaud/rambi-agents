import sys
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
import json
from typing import List, Sequence
from autogen_agentchat.messages import ChatMessage, StopMessage, TextMessage
from autogen_agentchat.teams import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.base import CancellationToken
from autogen_core.components.tools import FunctionTool
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_ext.models import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import BaseChatAgent
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import MaxMessageTermination
from autogen_ext.models import AzureOpenAIChatCompletionClient
from dataclasses import dataclass

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.DEBUG)

# Get configuration settings
load_dotenv()

@dataclass
class Movie:
    plot: str
    posterUrl: str 


class ImageAgent(ToolUseAssistantAgent):
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient) -> None:
        super().__init__(name=name, model_client=model_client, registered_tools=[
            FunctionTool(self.my_describe_movie_poster, description="Describe a movie poster based on the URL"),
        ], description="An agent that can describe images based on the URL, for example movie poster and other images.")

    async def my_describe_movie_poster(self, posterUrl: str) -> str: 
        print (f"\n----GPT4O describe_movie_poster called with {posterUrl}!!.\n")
        response = await self._model_client._client.chat.completions.create(
            model="gpt4o",
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
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
                ] } 
            ],
            max_tokens=2000 
        )
        # Return the generated description
        print (f"\n---GPT4O answer: {response.choices[0].message.content}.\n")
        return response.choices[0].message.content
    


class MovieDatabaseAgent(ToolUseAssistantAgent):
    def __init__(self,  name: str, model_client: AzureOpenAIChatCompletionClient) -> None:
        super().__init__(name=name, model_client=model_client, registered_tools=[
            FunctionTool(self.my_get_movie_plot, description="Get the plot of a movie using its title"),
        ], description="An agent that can search information about movies (plot, actors, posters, etc.)")

    async def my_get_movie_plot(self, title: str) -> Movie:
        print (f"\n----SELF get_movie_plot called with {title}!!.\n")
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

        #return json.dumps({"plot": plot,"posterURL": posterURL}, indent=4)
        return Movie(plot, posterURL)
    
    
    
       

class UserProxyAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        super().__init__(name, description)

    @property
    def produced_message_types(self) -> List[type[ChatMessage]]:
        return [TextMessage, StopMessage]

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> ChatMessage:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "\nPlease Provide a Movie Title: ")
        if "TERMINATE" in user_input:
            return StopMessage(content="User has terminated the conversation.", source=self.name)
        return TextMessage(content=user_input, source=self.name)

async def get_movie_plot(title: str) -> str:
    print (f"\n----get_movie_plot called with {title}!!.\n")
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
        posterURL = "https://image.tmdb.org/t/p/original/rhYJKOt6UrQq7JQgLyQcSWW5R86.jpg"
    else:
        plot = f"I'm sorry, I don't know that the {title} movie."
        posterURL = "xxxxx"

    return json.dumps({"plot": plot,"posterURL": posterURL}, indent=4)

async def describe_movie_poster(posterUrl: str) -> str:
    print (f"\n----describe_movie_poster called with {posterUrl}!!.\n")
    return f"The poster is a beautiful image of the movie."

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
        description="An agent that can search information about movies (plot, actors, posters, etc.)",
        system_message="You are a helpful AI assistant. Solve tasks using your tools.",
        registered_tools=[
            FunctionTool(get_movie_plot, description="Get the plot of a movie"),
        ],
    )

    movie_database_agent = MovieDatabaseAgent("movie_database_agent", model_client)
    image_agent = ImageAgent("movie_poster_agent", model_client)
    #description = await movie_database_agent.my_describe_movie_poster("https://image.tmdb.org/t/p/original/wV9e2y4myJ4KMFsyFfWYcUOawyK.jpg")
    #print(description)

    #sys.exit(100)

    movie_advisor = CodingAssistantAgent(
        "movie_advisor",
        model_client=model_client,
        description="Assist in creating a new movie plot based on 2 existing movies",
        system_message="You are a helpful assistant that can suggest a new movie plot for a user based on 2 existing movies but have no knowledge about existing movies.",
    )

    describe_image_agent = ToolUseAssistantAgent(
        "describe_image",
        model_client=model_client,
        description="Assist in describing any images provided as an URL",
        system_message="You are a helpful AI assistant. Solve tasks using your tools.",
        registered_tools=[
            FunctionTool(describe_movie_poster, description="Describe a movie poster based on the URL"),
        ],
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

    user_proxy = UserProxyAgent("askformovie", "Ask for movies only")
    #group_chat = SelectorGroupChat([ movie_database, describe_image_agent, movie_advisor], model_client=model_client)
    group_chat = RoundRobinGroupChat([ movie_database,describe_image_agent, summary_agent])
    group_chat2 = RoundRobinGroupChat([ movie_database_agent,image_agent, summary_agent])
    group_chat3 = SelectorGroupChat([ movie_database_agent,image_agent, summary_agent], model_client=model_client)
    result = await group_chat2.run(task="The 2 movies are Bambi and Avatar. Grab information about these movies (plot, posterUrl and a sharp description of the poster). Use movie_poster_agent to get description of all posters. Result in French. ", 
                                  termination_condition=StopMessageTermination())
    #print(result)
    # Access the messages attribute directly
    print("== "+ str(len(result.messages))+ " message ======= \n")
    for message in result.messages:
        print(message.content)
        print("-----\n")
    print("/== "+ str(len(result.messages))+ " message ======= \n")    


asyncio.run(main())