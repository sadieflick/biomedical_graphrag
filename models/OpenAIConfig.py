# langchain and env vars
from typing import Dict, Any
import os
from config.objects import Config
from langchain.chat_models import ChatOpenAI, init_chat_model
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from typing import ClassVar

class OpenAIModel(ChatOpenAI):
    """Configuration for OpenAI model."""

    default_config: ClassVar[dict] = {
        "model":"gpt-4o",
        "temperature":0,
        "max_tokens":None,
        "timeout":None,
        "max_retries":2
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
        ### Be sure to add any other params to your env variables
        }
    
    def __init__(self, settings: Dict[str, Any] = None):

        _ = load_dotenv(find_dotenv()) # read local .env file

        try:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Please add an OpenAI Key to project environment variables (.env)")
        except ValueError as e:
            print(e)
        
        if settings == None:
            settings = self.default_config
        settings.update(Config().llm)
        
        # Initialize model
        self = init_chat_model(**settings, streaming=True)