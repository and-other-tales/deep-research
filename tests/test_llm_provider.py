"""
Test for the multi-LLM provider functionality.
Tests that the LLM_PROVIDER environment variable correctly controls which LLM is used.
"""
import os
import pytest
from unittest.mock import patch

from othertales.deepresearch.configuration import LLMProvider
from othertales.deepresearch.utils import get_llm

# Import LLM classes if available for assertion tests
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class MockLLM:
    def invoke(self, input, config=None, **kwargs):
        return {"content": "This is a mock response for testing"}

# Test setting provider through environment variable
@pytest.mark.parametrize(
    "provider_env,expected_provider,expected_available",
    [
        ("openai", "openai", OPENAI_AVAILABLE),
        ("anthropic", "anthropic", ANTHROPIC_AVAILABLE),
        ("invalid-provider", "", False),  # Should default to empty string for auto-detection
    ],
)
def test_provider_from_env(provider_env, expected_provider, expected_available):
    """Test that the LLM provider is correctly set from the environment variable."""
    # Save current environment
    old_env = os.environ.get("LLM_PROVIDER")
    
    try:
        # Set environment variable
        os.environ["LLM_PROVIDER"] = provider_env
        
        # Mock API keys to avoid actual API calls
        mock_env = {
            "OPENAI_API_KEY": "sk-mock-key" if expected_provider == "openai" else None,
            "ANTHROPIC_API_KEY": "sk-mock-key" if expected_provider == "anthropic" else None,
        }
        
        # Apply mock environment
        with patch.dict(os.environ, mock_env):
            # Get LLM
            llm = get_llm()
            
            # Check if we got the expected provider
            if expected_provider == "openai" and expected_available:
                assert isinstance(llm, ChatOpenAI)
            elif expected_provider == "anthropic" and expected_available:
                assert isinstance(llm, ChatAnthropic)
            else:
                # For invalid providers or unavailable implementations, we expect a mock LLM
                # Just check that it has an invoke method
                assert hasattr(llm, "invoke")
    finally:
        # Restore original environment
        if old_env is not None:
            os.environ["LLM_PROVIDER"] = old_env
        else:
            os.environ.pop("LLM_PROVIDER", None)

# Test setting provider directly in the get_llm function
@pytest.mark.parametrize(
    "provider_arg,expected_available",
    [
        ("openai", OPENAI_AVAILABLE),
        ("anthropic", ANTHROPIC_AVAILABLE),
        ("invalid-provider", False),  # Should default to empty string for auto-detection
    ],
)
def test_provider_from_argument(provider_arg, expected_available):
    """Test that the LLM provider is correctly set from the function argument."""
    # Mock API keys to avoid actual API calls
    mock_env = {
        "OPENAI_API_KEY": "sk-mock-key" if provider_arg == "openai" else None,
        "ANTHROPIC_API_KEY": "sk-mock-key" if provider_arg == "anthropic" else None,
    }
    
    # Apply mock environment
    with patch.dict(os.environ, mock_env):
        # Get LLM with explicit provider
        llm = get_llm(provider=provider_arg)
        
        # Check if we got the expected provider
        if provider_arg == "openai" and expected_available:
            assert isinstance(llm, ChatOpenAI)
        elif provider_arg == "anthropic" and expected_available:
            assert isinstance(llm, ChatAnthropic)
        else:
            # For invalid providers or unavailable implementations, we expect a mock LLM
            # Just check that it has an invoke method
            assert hasattr(llm, "invoke")

# Test auto-detection based on available API keys
def test_provider_auto_detection():
    """Test that the LLM provider is correctly auto-detected from available API keys."""
    # Save current environment
    old_llm_provider = os.environ.get("LLM_PROVIDER")
    old_openai_key = os.environ.get("OPENAI_API_KEY")
    old_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    try:
        # Clear LLM_PROVIDER to force auto-detection
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        
        # Test OpenAI priority when both keys are available
        mock_env = {
            "OPENAI_API_KEY": "sk-mock-key",
            "ANTHROPIC_API_KEY": "sk-mock-key",
        }
        
        with patch.dict(os.environ, mock_env):
            llm = get_llm()
            if OPENAI_AVAILABLE:
                assert isinstance(llm, ChatOpenAI)
            else:
                assert hasattr(llm, "invoke")
        
        # Test Anthropic when only Anthropic key is available
        mock_env = {
            "OPENAI_API_KEY": None,
            "ANTHROPIC_API_KEY": "sk-mock-key",
        }
        
        with patch.dict(os.environ, mock_env):
            llm = get_llm()
            if ANTHROPIC_AVAILABLE:
                assert isinstance(llm, ChatAnthropic)
            else:
                assert hasattr(llm, "invoke")
        
        # Test mock LLM when no keys are available
        mock_env = {
            "OPENAI_API_KEY": None,
            "ANTHROPIC_API_KEY": None,
        }
        
        with patch.dict(os.environ, mock_env):
            llm = get_llm()
            # Should be a mock LLM with an invoke method
            assert hasattr(llm, "invoke")
            
    finally:
        # Restore original environment
        if old_llm_provider is not None:
            os.environ["LLM_PROVIDER"] = old_llm_provider
        elif "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
            
        if old_openai_key is not None:
            os.environ["OPENAI_API_KEY"] = old_openai_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            
        if old_anthropic_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_anthropic_key
        elif "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]