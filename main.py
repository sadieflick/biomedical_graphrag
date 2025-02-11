# src/main.py
import sys
import logging
from config.config import Config
from pathlib import Path
from neo4j import GraphDatabase
from src.llm_setup import initialize_meditron

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BiomedicalGraphRAG:
    def __init__(self):
        self.config = Config()
        self.llm = None
        self.neo4j_driver = None
        
    def initialize(self):
        """Initialize all components of the system"""
        logger.info("Initializing BiomedicalGraphRAG system...")
        
        try:
            # Initialize Neo4j connection
            logger.info("Connecting to Neo4j...")
            self.neo4j_driver = GraphDatabase.driver(
                self.config.neo4j['uri'],
                auth=(self.config.neo4j['user'], self.config.neo4j['password'])
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Neo4j connection established successfully")
            
            # Initialize LLM
            logger.info("Initializing LLM...")
            self.llm = initialize_meditron(self.config.llm)
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

def main():
    # Initialize the system
    system = BiomedicalGraphRAG()
    
    if not system.initialize():
        logger.error("Failed to initialize the system")
        sys.exit(1)
    
    if not system.test_system():
        logger.error("System tests failed")
        system.cleanup()
        sys.exit(1)
    
    logger.info("System initialized and tested successfully")
    
    # Keep system running for interactive use
    try:
        while True:
            query = input("\nEnter a query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            response = system.llm(query)
            print("\nResponse:", response)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()