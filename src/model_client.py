from openai import AsyncAzureOpenAI, AsyncOpenAI
import os
from constant import *
from dotenv import load_dotenv

load_dotenv()

if PROVIDER == "OPENAI":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = AsyncOpenAI(api_key=openai_api_key)

elif PROVIDER == "AZURE":
    openai_client = AsyncAzureOpenAI(
      azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
      api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      api_version="2024-02-01"
    )


