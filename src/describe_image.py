import asyncio
import base64
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
# Get configuration settings
load_dotenv()
from openai import AzureOpenAI

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key= os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = 'gpt4o'
api_version = '2024-08-01-preview' # this might change in the future

client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)

response = client.chat.completions.create(
    model=deployment_name,
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
                    "url": "https://image.tmdb.org/t/p/original/wV9e2y4myJ4KMFsyFfWYcUOawyK.jpg"
                }
            }
        ] } 
    ],
    max_tokens=2000 
)

print(response)
print("\n------n")
print(response.choices[0].message.content)