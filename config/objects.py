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
            'model': os.getenv('MODEL'),
            'temperature': 0.0,
            'max_tokens': 2000,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'max_retries': 2,

        }

        print(f'\n\n\n=========== SETTINGS IN CONFIG OBJ: {self.llm} ==============\n\n\n')

