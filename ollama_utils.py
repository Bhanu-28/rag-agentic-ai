"""
Ollama utilities for local LLM and embeddings.

Provides LangChain-compatible wrappers for Ollama models as drop-in
replacements for cloud-based APIs (IBM Watson, OpenAI, etc.)
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


def get_ollama_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.5,
    **kwargs
):
    """
    Get an Ollama LLM instance compatible with LangChain.
    
    Args:
        model: Model name (default: from env or qwen2.5-coder:3b)
        base_url: Ollama server URL (default: from env or localhost:11434)
        temperature: Sampling temperature
        **kwargs: Additional parameters for ChatOllama
        
    Returns:
        ChatOllama instance
        
    Example:
        >>> llm = get_ollama_llm()
        >>> response = llm.invoke("Explain Python decorators")
        >>> print(response.content)
    """
    from langchain_ollama import ChatOllama
    
    return ChatOllama(
        model=model or os.getenv("OLLAMA_LLM_MODEL", "qwen2.5-coder:3b"),
        base_url=base_url or os.getenv("OLLAMA_LLM_URL", "http://localhost:11434"),
        temperature=temperature,
        **kwargs
    )


def get_ollama_embeddings(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
):
    """
    Get Ollama embeddings instance compatible with LangChain.
    
    Args:
        model: Embedding model name (default: from env or nomic-embed-text:v1.5)
        base_url: Ollama server URL (default: from env or localhost:11434)
        **kwargs: Additional parameters for OllamaEmbeddings
        
    Returns:
        OllamaEmbeddings instance
        
    Example:
        >>> embeddings = get_ollama_embeddings()
        >>> vectors = embeddings.embed_documents(["Hello", "World"])
        >>> print(f"Dimension: {len(vectors[0])}")
    """
    from langchain_ollama import OllamaEmbeddings
    
    return OllamaEmbeddings(
        model=model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5"),
        base_url=base_url or os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434"),
        **kwargs
    )


def llm_model(prompt_txt: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Drop-in replacement for IBM Watson llm_model function.
    
    Args:
        prompt_txt: The prompt to send to the model
        params: Optional parameters (temperature, max_tokens, etc.)
        
    Returns:
        Model response as string
        
    Example:
        >>> response = llm_model("What is Python?")
        >>> print(response)
    """
    default_params = {
        "temperature": 0.5,
    }
    
    if params:
        default_params.update(params)
    
    llm = get_ollama_llm(**default_params)
    response = llm.invoke(prompt_txt)
    return response.content


# Quick test
if __name__ == "__main__":
    print("Testing Ollama LLM...")
    try:
        llm = get_ollama_llm()
        response = llm.invoke("Say 'Hello' in one word")
        print(f"✓ LLM Response: {response.content}")
    except Exception as e:
        print(f"✗ LLM Error: {e}")
    
    print("\nTesting Ollama Embeddings...")
    try:
        embed = get_ollama_embeddings()
        vectors = embed.embed_documents(["test query"])
        print(f"✓ Embedding dimension: {len(vectors[0])}")
    except Exception as e:
        print(f"✗ Embedding Error: {e}")
