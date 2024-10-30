

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
    UserMessage,
)
from dotenv import load_dotenv
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.logging import ConsoleLogHandler

# Get configuration settings 
load_dotenv()

# Configure logging
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.DEBUG)

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
    
    # Log the request details
    logging.debug("Sending request to Azure OpenAI Chat Completion API")
  
    model_client_result = await model_client.create(
        messages=[
            UserMessage(content="What is the capital of France?", source="user"),
        ]
    )
    print(model_client_result.content)  # "Paris"

asyncio.run(main())