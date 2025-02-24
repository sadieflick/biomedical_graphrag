import sys
import logging
import chainlit as cl
from pathlib import Path
from config.objects import Config
from models.OpenAIConfig import OpenAIModel
from typing import cast
from neo4j import GraphDatabase
from src.llm_setup import *
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiomedicalGraphRAG:
    def __init__(self):
        print(type(objects))
        self.settings = Config()
        self.llm = initialize_model()
        self.neo4j_driver = None
        
        
    def initialize(self):
        """Initialize all components of the system"""
        logger.info("Initializing BiomedicalGraphRAG system...")
        
        try:
            # Initialize Neo4j connection
            logger.info("Connecting to Neo4j...")
            self.neo4j_driver = GraphDatabase.driver(
                self.settings.neo4j['uri'],
                auth=(self.settings.neo4j['user'], self.settings.neo4j['password'])
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection established successfully")
            
            # Initialize LLM
            logger.info("Initializing LLM...")
            self.llm = OpenAIConfig.OpenAIModel(self.settings)
            logger.info("LLM initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def test_system(self):
        """Run basic system tests"""
        try:
            # Test Neo4j
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1 as num")
                assert result.single()["num"] == 1
                logger.info("Neo4j test query successful")
            
            # Test LLM
            test_response = self.llm("What is the GWAS p-value for the association between childhood-onset asthma and RORA?")
            assert test_response is not None
            logger.info("LLM test query successful")
            
            return True
            
        except Exception as e:
            logger.error(f"System test failed: {str(e)}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
        logger.info("Cleanup completed")


print(f'************{__name__}*****************')
@cl.on_chat_start
async def on_chat_start():
    system = BiomedicalGraphRAG()
    model = system.llm
    runnable = create_biomedical_chain(model)
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"query": message.content, "graph_context": None},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

# if __name__ == "__main__":
#     main()