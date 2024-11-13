from PIL import Image
import requests
import asyncio
import base64
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import os
# Get configuration settings
load_dotenv()

api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = 'gpt4o'
api_version = '2024-08-01-preview'  # this might change in the future

try:
    client = AzureOpenAI()

    result = client.images.generate(
        model="dall-e-3",  # the name of your DALL-E 3 deployment
        prompt="The movie poster captures the heartwarming and adventurous spirit of the film, showcasing Bambi and his animal band clad in mini black suits and hats, instruments in paw. They stand on a makeshift stage set amidst the vibrant greenery of the forest, where magical musical notes float around, captivating their friends and enemies alike. The backdrop intertwines elements of the serene forest with the lively, electrifying vibe of a rock concert.!!",
        n=1,
        size='1024x1792'
    )

    json_response = json.loads(result.model_dump_json())

    print(json_response)
    # Retrieve the generated image
    # extract image URL from response
    image_url = json_response["data"][0]["url"]
    print(f"=> image URL {image_url}")
except Exception as e:
    print(f"Generation Image Error: {e}")
