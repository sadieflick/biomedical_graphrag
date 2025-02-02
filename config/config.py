import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        
        self.neo4j = {
            'uri': os.getenv('NEO4J_URI'),
            'user': os.getenv('NEO4J_USER'),
            'password': os.getenv('NEO4J_PASSWORD'),
        }
        
        self.llm = {
            'model_path': os.getenv('MODEL_PATH'),
            'temperature': 0.75,
            'max_tokens': 2000,
        }