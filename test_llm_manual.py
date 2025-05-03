#!/usr/bin/env python
"""
Manual test for the multi-LLM provider functionality.
This script demonstrates how to use the LLM_PROVIDER environment variable.
"""
import os
import sys
from open_deep_research.utils import get_llm
from open_deep_research.configuration import LLMProvider

def main():
    # Show available providers
    print("Available providers in LLMProvider enum:")
    for provider in LLMProvider:
        print(f"  - {provider.name}: {provider.value}")
    print()
    
    # First test: current environment setting
    current_provider = os.environ.get("LLM_PROVIDER", "[not set]")
    print(f"Current LLM_PROVIDER from environment: {current_provider}")
    
    # Get LLM with current environment
    llm = get_llm()
    print(f"Current LLM instance: {llm.__class__.__name__}")
    print()
    
    # Explicitly try different providers
    providers_to_test = [
        "openai",
        "anthropic",
        "google",
        "huggingface"
    ]
    
    for provider in providers_to_test:
        print(f"Testing provider: {provider}")
        llm = get_llm(provider=provider)
        print(f"  LLM instance: {llm.__class__.__name__}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()