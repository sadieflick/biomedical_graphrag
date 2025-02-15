# langchain and env vars
from typing import Dict, Any
import os, openai
from config.config import Config
from langchain.chat_models import ChatOpenAI, init_chat_model
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

class OpenAIModel:
    """Configuration for OpenAI model."""

    default_config = {
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
    
    def __init__(self, config: Dict[str, Any] = None):

        _ = load_dotenv(find_dotenv()) # read local .env file

        try:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Please add an OpenAI Key to project environment variables (.env)")
        except ValueError as e:
            print(e)
        
        
        openai.api_key = os.environ['OPENAI_API_KEY']

        
        config = self.default_config
        config.update(Config().llm)
        
        # Initialize model
        model = init_chat_model(**config)
        
        return model