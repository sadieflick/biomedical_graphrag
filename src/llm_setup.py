# src/llm_setup.py
from typing import Dict, Any
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class MeditronConfig:
    """Configuration for Meditron model."""
    def __init__(self):
        self.model_config = {
            # Model parameters
            "model_path": "./models/meditron-7b.Q4_K_M.gguf",  # Quantized model path
            "n_gpu_layers": 16,  # Adjust based on your GPU memory
            "n_batch": 512,     # Batch size for prompt processing
            "n_ctx": 4096,      # Context window size
            
            # Generation parameters
            "temperature": 0.7,  # Lower for more focused medical responses
            "max_tokens": 2048,  # Maximum generation length
            "top_p": 0.9,       # Nucleus sampling
            "top_k": 40,        # Top-k sampling
            "repeat_penalty": 1.1,  # Helps prevent repetitive text
            
            # Performance options
            "f16_kv": True,     # Use float16 for key/value cache
            "vocab_only": False, # Load only vocabulary
            "use_mlock": True,  # Lock memory to prevent swapping
            "use_mmap": True,   # Use memory mapping for faster loading
            
            # Threading options
            "n_threads": None,   # Will use all available threads if None
            "n_threads_batch": None  # Threads to use for batch processing
        }

def initialize_meditron(config: Dict[str, Any] = None) -> LlamaCpp:
    """
    Initialize Meditron model with specified configuration.
    
    Args:
        config: Optional configuration override
        
    Returns:
        LlamaCpp: Configured Meditron model instance
    """
    if config is None:
        config = MeditronConfig().model_config
        
    # Setup streaming callback
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Initialize model
    llm = LlamaCpp(
        callback_manager=callback_manager,
        verbose=True,
        **config
    )
    
    return llm

def create_biomedical_chain():
    """
    Create a specialized chain for biomedical queries.
    """
    llm = initialize_meditron()
    
    # Specialized prompt template for biomedical queries
    bio_template = """You are a biomedical AI assistant using the Meditron model, 
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
    
    return LLMChain(llm=llm, prompt=prompt)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the current model configuration.
    """
    config = MeditronConfig().model_config
    return {
        "model_name": "Meditron-7B",
        "context_length": config["n_ctx"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "gpu_layers": config["n_gpu_layers"]
    }

# Example usage and testing function
def test_meditron_response(query: str) -> str:
    """
    Test Meditron's response to a biomedical query.
    
    Args:
        query: Biomedical query string
        
    Returns:
        str: Model's response
    """
    llm = initialize_meditron()
    response = llm(f"Query: {query}\nResponse:")
    return response