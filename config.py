"""
Configuration module for loading environment variables and API keys.

This module provides a centralized way to access all configuration settings
and API keys from environment variables.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class to access environment variables."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")

    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPEN_ROUTER_API_KEY")
    
    # Anthropic (Claude) Configuration
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    
    # Google AI (Gemini) Configuration
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-pro")
    
    # Cohere Configuration
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    
    # Hugging Face Configuration
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    HUGGINGFACE_MODEL: Optional[str] = os.getenv("HUGGINGFACE_MODEL")
    
    # Pinecone Vector Database Configuration
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: Optional[str] = os.getenv("PINECONE_INDEX_NAME")
    
    # Weaviate Vector Database Configuration
    WEAVIATE_URL: Optional[str] = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY: Optional[str] = os.getenv("WEAVIATE_API_KEY")
    
    # Qdrant Vector Database Configuration
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # LangSmith Configuration
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: Optional[str] = os.getenv("LANGSMITH_PROJECT")
    
    # Application Settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate_keys(cls, required_keys: list[str]) -> bool:
        """
        Validate that required API keys are present.
        
        Args:
            required_keys: List of required configuration keys
            
        Returns:
            True if all required keys are present, False otherwise
            
        Raises:
            ValueError: If any required key is missing
        """
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key, None):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(
                f"Missing required API keys: {', '.join(missing_keys)}. "
                f"Please set them in your .env file."
            )
        return True


# Example usage:
if __name__ == "__main__":
    # Validate OpenAI key is present (example)
    try:
        Config.validate_keys(["OPENAI_API_KEY"])
        print("✓ All required API keys are configured")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
