import pytest
from neo4j import GraphDatabase
from src.llm_setup import initialize_llm
from config.config import Config

def test_neo4j_connection():
    config = Config()
    with GraphDatabase.driver(**config.neo4j) as driver:
        result = driver.verify_connectivity()
        assert result is not None

def test_llm_initialization():
    config = Config()
    llm = initialize_llm(config.llm['model_path'])
    response = llm("Test query: What is DNA?")
    assert response is not None