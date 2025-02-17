# src/llm_setup.py
import os
import config.objects as objects
from typing import Dict, Any
from models import OpenAIConfig
from config.objects import Config
from langchain.schema import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv


def initialize_model():
    """
    Args:
    Returns:
        model instance
    """
    llm_settings = Config().llm
    llm = ChatOpenAI(**llm_settings)
    # try:
    #     # llm_settings = Config().llm
    #     # llm = None
    #     # if os.environ.get('MODEL_TYPE') == 'OpenAI':
    #     # llm = OpenAIConfig.OpenAIModel(llm_settings)
    #     # print(f"**************llm**************\n{llm}")
        
    #     # TO DO: Add other model logic as needed
    #     # raise(Exception('Something went wrong with instantiating the model. Be sure the model type is supported, and that the environment variable MODEL_TYPE exits.'))
    # except Exception as err:
    #     print(f"Unexpected {err=}, {type(err)=}")
    
    return llm

def create_biomedical_chain(llm):
    """
    Create a specialized chain for biomedical queries.
    """
    
    # Specialized prompt template for biomedical queries
    bio_template = """You are a biomedical expert, 
    specifically trained for medical and scientific analysis. Based on the provided 
    context and your knowledge, please address the following query:

    Context from knowledge graph: {graph_context}
    
    Query: {query}
    
    Please provide a detailed, scientifically accurate response while maintaining clarity. 
    Include relevant biological mechanisms and cite any specific relationships from 
    the provided context.

    Response:"""
    
    prompt = PromptTemplate(
        template=bio_template,
        input_variables=["graph_context", "query"]
    )
    
    return prompt | llm | StrOutputParser

def get_model_info():
    """
    Get information about the current model configuration.
    """
    llm_config = objects().llm
    return llm_config

# Example usage and testing function
def test_response(query: str) -> str:
    """
    Test Meditron's response to a biomedical query.
    
    Args:
        query: Biomedical query string
        
    Returns:
        str: Model's response
    """
    llm = initialize_model()
    response = llm(f"Query: {query}\nResponse:")
    return response