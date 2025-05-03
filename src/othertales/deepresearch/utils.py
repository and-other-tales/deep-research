import os
import asyncio
import requests
import random 
import concurrent
import aiohttp
import httpx
import sys
import logging
import time
from typing import List, Optional, Dict, Any, Union, Literal
from urllib.parse import unquote, urlparse

from exa_py import Exa
from linkup import LinkupClient
from tavily import AsyncTavilyClient
from duckduckgo_search import DDGS 
from bs4 import BeautifulSoup
from markdownify import markdownify

from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_core.tools import tool
from langchain_core.runnables import Runnable

# LLM providers imports
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

try:
    from langchain_openai import AzureChatOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

try:
    from langchain_aws import ChatBedrockConverse
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from langsmith import traceable

from othertales.deepresearch.state import Section
from othertales.deepresearch.configuration import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_llm(provider: str = None, model: str = None, temperature: float = 0.2, **kwargs) -> Runnable:
    """Get a language model based on the provider and model specified.
    
    This function supports multiple LLM providers based on environment variables:
    - LLM_PROVIDER: Set to one of the following values to specify the provider:
      - "openai": Use OpenAI models
      - "anthropic": Use Anthropic Claude models
      - "azure": Use Azure OpenAI models
      - "bedrock": Use AWS Bedrock models
      - "huggingface": Use HuggingFace pipeline models (local models)
      - "huggingface-endpoint": Use HuggingFace endpoint models (API-based)
      - "google": Use Google Gemini models

    Provider-specific configurations from environment variables:
    
    OpenAI configuration:
    - OPENAI_API_KEY: API key (required)
    - OPENAI_MODEL: Model name (default depends on the model parameter)
    - OPENAI_BASE_URL: Optional base URL for API endpoint
    - OPENAI_ORG_ID: Optional organization ID
    
    Anthropic Claude configuration:
    - ANTHROPIC_API_KEY: API key (required)
    - ANTHROPIC_MODEL: Model name (default depends on the model parameter)
    
    Azure OpenAI configuration:
    - AZURE_OPENAI_API_KEY: API key (required)
    - AZURE_OPENAI_ENDPOINT: Azure endpoint URL (required)
    - AZURE_OPENAI_DEPLOYMENT_NAME: Deployment name (required)
    - AZURE_OPENAI_API_VERSION: API version (default: "2023-12-01-preview")
    
    AWS Bedrock configuration:
    - BEDROCK_MODEL_ID: Model ID (default: "anthropic.claude-3-sonnet-20240229-v1:0")
    - AWS_REGION: AWS region (default: "us-east-1")
    
    HuggingFace Pipeline configuration (for local models):
    - HUGGINGFACE_MODEL_ID: Model ID
    - HUGGINGFACE_DEVICE: Device to run on (default: -1 for CPU, use 0 for first GPU)
    - HUGGINGFACE_MAX_NEW_TOKENS: Max new tokens for generation (default: 512)
    - HUGGINGFACE_TEMPERATURE: Temperature for generation
    - HUGGINGFACE_TOP_K: Top-k for sampling (default: 50)
    - HUGGINGFACE_TOP_P: Top-p for sampling (default: 0.95)
    
    HuggingFace Endpoint configuration (for API-based models):
    - HUGGINGFACE_ENDPOINT_URL: Endpoint URL (required)
    - HUGGINGFACE_ENDPOINT_TOKEN: HuggingFace API token (required)
    - HUGGINGFACE_MAX_NEW_TOKENS: Max new tokens for generation (default: 512)
    - HUGGINGFACE_TEMPERATURE: Temperature for generation
    
    Google Gemini configuration:
    - GOOGLE_API_KEY: API key (required)
    - GOOGLE_MODEL: Model name (default: "gemini-1.5-pro")
    - GOOGLE_TEMPERATURE: Temperature for generation
    
    Args:
        provider (str, optional): The LLM provider to use. Defaults to environment variable LLM_PROVIDER or auto-detection.
        model (str, optional): The specific model to use. If not provided, will use default for the provider.
        temperature (float, optional): Temperature setting for generation. Defaults to 0.2.
        **kwargs: Additional arguments to pass to the model.
        
    Returns:
        Runnable: A configured LLM instance
        
    Raises:
        ImportError: If the required package for the selected provider is not installed
        ValueError: If required configuration for the provider is missing
    """
    # Define a mock LLM for testing or when no valid provider is available
    class MockLLM(Runnable):
        """Mock LLM for testing."""
        
        def __init__(self):
            pass
            
        def invoke(self, input, config=None, **kwargs):
            """Mock invoke method."""
            return {"content": "This is a mock response for testing"}
        
        async def ainvoke(self, input, config=None, **kwargs):
            """Mock async invoke method."""
            return {"content": "This is a mock async response for testing"}
            
        def _call(self, input):
            """Legacy _call method."""
            return "This is a mock response for testing"
            
        def _acall(self, input):
            """Legacy _acall method."""
            return "This is a mock async response for testing"
    
    # Detect if we're running in a test
    is_test = "pytest" in sys.modules or "unittest" in sys.modules
    
    # If provider is not specified, get from environment variable
    if not provider:
        provider = os.environ.get("LLM_PROVIDER", "").lower()
    else:
        provider = provider.lower()
    
    # Validate provider if specified
    valid_providers = [
        "openai", "anthropic", "azure", "bedrock", 
        "huggingface", "huggingface-endpoint", "google"
    ]
    
    if provider and provider not in valid_providers:
        logger.warning(f"Invalid provider value: '{provider}'. Using auto-detection.")
        provider = ""
    
    # If no provider specified or invalid, auto-detect based on available API keys
    if not provider:
        if os.environ.get("OPENAI_API_KEY") and OPENAI_AVAILABLE:
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY") and ANTHROPIC_AVAILABLE:
            provider = "anthropic"
        elif os.environ.get("AZURE_OPENAI_API_KEY") and AZURE_OPENAI_AVAILABLE:
            provider = "azure"
        elif os.environ.get("GOOGLE_API_KEY") and GOOGLE_AVAILABLE:
            provider = "google"
        elif (os.environ.get("HUGGINGFACE_ENDPOINT_URL") and 
              os.environ.get("HUGGINGFACE_ENDPOINT_TOKEN") and HUGGINGFACE_AVAILABLE):
            provider = "huggingface-endpoint"
        elif os.environ.get("HUGGINGFACE_MODEL_ID") and HUGGINGFACE_AVAILABLE:
            provider = "huggingface"
        elif BEDROCK_AVAILABLE and os.environ.get("AWS_REGION"):
            provider = "bedrock"
        else:
            logger.warning("No LLM provider detected. Using mock LLM.")
            return MockLLM()
    
    try:
        # 1. OpenAI models
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("langchain_openai is not installed. Install with: pip install langchain-openai")
            
            model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            org_id = os.environ.get("OPENAI_ORG_ID")
            
            if not api_key:
                logger.warning("OpenAI API key not found, using mock LLM")
                return MockLLM()
                
            logger.info(f"Using OpenAI with model {model_name}")
            
            # Initialize with optional parameters if provided
            llm_kwargs = {
                "model": model_name,
                "temperature": temperature,
                "api_key": api_key,
                **kwargs
            }
            
            if base_url:
                llm_kwargs["base_url"] = base_url
            if org_id:
                llm_kwargs["organization"] = org_id
                
            return ChatOpenAI(**llm_kwargs)
        
        # 2. Anthropic Claude models
        elif provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("langchain_anthropic is not installed. Install with: pip install langchain-anthropic")
            
            model_name = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-latest")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            
            if not api_key:
                logger.warning("Anthropic API key not found, using mock LLM")
                return MockLLM()
                
            logger.info(f"Using Anthropic Claude with model {model_name}")
            
            return ChatAnthropic(
                model=model_name, 
                temperature=temperature, 
                anthropic_api_key=api_key,
                **kwargs
            )
        
        # 3. Azure OpenAI models
        elif provider == "azure":
            if not AZURE_OPENAI_AVAILABLE:
                raise ImportError("langchain_openai is not installed. Install with: pip install langchain-openai")
            
            deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
            
            if not all([deployment_name, endpoint, api_key]):
                logger.warning("Azure OpenAI configuration incomplete, using mock LLM")
                return MockLLM()
                
            logger.info(f"Using Azure OpenAI with deployment {deployment_name}")
            
            return AzureChatOpenAI(
                deployment_name=deployment_name,
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                temperature=temperature,
                **kwargs
            )
        
        # 4. AWS Bedrock models
        elif provider == "bedrock":
            if not BEDROCK_AVAILABLE:
                raise ImportError("langchain_aws is not installed. Install with: pip install langchain-aws")
                
            model_id = model or os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
            region = os.environ.get("AWS_REGION", "us-east-1")
            
            logger.info(f"Using AWS Bedrock with model {model_id}")
            
            return ChatBedrockConverse(
                model_id=model_id, 
                region_name=region, 
                temperature=temperature,
                **kwargs
            )
        
        # 5. HuggingFace Pipeline models (local models)
        elif provider == "huggingface":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("langchain_huggingface is not installed. Install with: pip install langchain-huggingface")
                
            model_id = model or os.environ.get("HUGGINGFACE_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
            device = int(os.environ.get("HUGGINGFACE_DEVICE", "-1"))  # Default to CPU with -1
            max_new_tokens = int(os.environ.get("HUGGINGFACE_MAX_NEW_TOKENS", "512"))
            hf_temperature = float(os.environ.get("HUGGINGFACE_TEMPERATURE", str(temperature)))
            top_k = int(os.environ.get("HUGGINGFACE_TOP_K", "50"))
            top_p = float(os.environ.get("HUGGINGFACE_TOP_P", "0.95"))
            
            logger.info(f"Loading HuggingFace Pipeline model {model_id} on device {device}")
            
            try:
                # Create HuggingFace Pipeline LLM
                hf_pipeline = HuggingFacePipeline.from_model_id(
                    model_id=model_id,
                    task="text-generation",
                    device=device,
                    pipeline_kwargs={
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": hf_temperature,
                        "top_k": top_k,
                        "top_p": top_p,
                        **kwargs
                    },
                )
                # Create ChatHuggingFace using the pipeline
                return ChatHuggingFace(llm=hf_pipeline)
            except Exception as e:
                logger.error(f"Error initializing HuggingFace Pipeline model: {e}")
                return MockLLM()
                
        # 6. HuggingFace Endpoint models (API-based)
        elif provider == "huggingface-endpoint":
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("langchain_huggingface is not installed. Install with: pip install langchain-huggingface")
            
            endpoint_url = os.environ.get("HUGGINGFACE_ENDPOINT_URL")
            hf_token = os.environ.get("HUGGINGFACE_ENDPOINT_TOKEN")
            max_new_tokens = int(os.environ.get("HUGGINGFACE_MAX_NEW_TOKENS", "512"))
            hf_temperature = float(os.environ.get("HUGGINGFACE_TEMPERATURE", str(temperature)))
            
            if not all([endpoint_url, hf_token]):
                logger.warning("HuggingFace Endpoint configuration incomplete, using mock LLM")
                return MockLLM()
                
            logger.info(f"Using HuggingFace Endpoint {endpoint_url}")
            
            # Create HuggingFace Endpoint
            hf_endpoint = HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=hf_token,
                task="text-generation",
                max_new_tokens=max_new_tokens,
                temperature=hf_temperature,
                **kwargs
            )
            
            # Wrap with ChatHuggingFace
            return ChatHuggingFace(llm=hf_endpoint)
            
        # 7. Google Gemini models
        elif provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("langchain_google_genai is not installed. Install with: pip install langchain-google-genai")
                
            model_name = model or os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")
            api_key = os.environ.get("GOOGLE_API_KEY")
            google_temperature = float(os.environ.get("GOOGLE_TEMPERATURE", str(temperature)))
            
            if not api_key:
                logger.warning("Google API key not found, using mock LLM")
                return MockLLM()
                
            logger.info(f"Using Google Gemini with model {model_name}")
            
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=google_temperature,
                google_api_key=api_key,
                **kwargs
            )
    
    except Exception as e:
        logger.warning(f"Error initializing LLM: {e}. Using mock LLM")
        return MockLLM()
    
def get_config_value(value):
    """
    Helper function to handle string, dict, and enum cases of configuration values
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filters the search_api_config dictionary to include only parameters accepted by the specified search API.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): The configuration dictionary for the search API.

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # Define accepted parameters for each search API
    SEARCH_API_PARAMS = {
        "exa": ["max_characters", "num_results", "include_domains", "exclude_domains", "subpages"],
        "tavily": ["max_results", "topic"],
        "perplexity": [],  # Perplexity accepts no additional parameters
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "linkup": ["depth"],
        "direct_url": ["use_playwright", "remove_selectors", "wait_until", "wait_for_selector", "extract_selectors"],
    }

    # Get the list of accepted parameters for the given search API
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # If no config provided, return an empty dict
    if not search_api_config:
        return {}

    # Filter the config to only include accepted parameters
    return {k: v for k, v in search_api_config.items() if k in accepted_params}

def deduplicate_and_format_sources(search_response, max_tokens_per_source=5000, include_raw_content=True):
    """
    Takes a list of search responses and formats them into a readable string.
    Limits the raw_content to approximately max_tokens_per_source tokens.
 
    Args:
        search_responses: List of search response dicts, each containing:
            - query: str
            - results: List of dicts with fields:
                - title: str
                - url: str
                - content: str
                - score: float
                - raw_content: str|None
        max_tokens_per_source: int
        include_raw_content: bool
            
    Returns:
        str: Formatted string with deduplicated sources
    """
     # Collect all results
    sources_list = []
    for response in search_response:
        sources_list.extend(response['results'])
    
    # Deduplicate by URL
    unique_sources = {source['url']: source for source in sources_list}

    # Format output
    formatted_text = "Content from sources:\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"{'='*80}\n"  # Clear section separator
        formatted_text += f"Source: {source['title']}\n"
        formatted_text += f"{'-'*80}\n"  # Subsection separator
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
        formatted_text += f"{'='*80}\n\n" # End section separator
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

@traceable
async def tavily_search_async(search_queries, max_results: int = 5, topic: str = "general", include_raw_content: bool = True):
    """
    Performs concurrent web searches with the Tavily API

    Args:
        search_queries (List[str]): List of search queries to process

    Returns:
            List[dict]: List of search responses from Tavily API:
                {
                    'query': str,
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

@traceable
def perplexity_search(search_queries):
    """Search the web using the Perplexity API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
  
    Returns:
        List[dict]: List of search responses from Perplexity API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    search_docs = []
    for query in search_queries:

        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "Search the web and provide factual information with sources."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse the response
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        citations = data.get("citations", ["https://perplexity.ai"])
        
        # Create results list for this query
        results = []
        
        # First citation gets the full content
        results.append({
            "title": f"Perplexity Search, Source 1",
            "url": citations[0],
            "content": content,
            "raw_content": content,
            "score": 1.0  # Adding score to match Tavily format
        })
        
        # Add additional citations without duplicating content
        for i, citation in enumerate(citations[1:], start=2):
            results.append({
                "title": f"Perplexity Search, Source {i}",
                "url": citation,
                "content": "See primary source for full content",
                "raw_content": None,
                "score": 0.5  # Lower score for secondary sources
            })
        
        # Format response to match Tavily structure
        search_docs.append({
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": [],
            "results": results
        })
    
    return search_docs

@traceable
async def exa_search(search_queries, max_characters: Optional[int] = None, num_results=5, 
                     include_domains: Optional[List[str]] = None, 
                     exclude_domains: Optional[List[str]] = None,
                     subpages: Optional[int] = None):
    """Search the web using the Exa API.
    
    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        max_characters (int, optional): Maximum number of characters to retrieve for each result's raw content.
                                       If None, the text parameter will be set to True instead of an object.
        num_results (int): Number of search results per query. Defaults to 5.
        include_domains (List[str], optional): List of domains to include in search results. 
            When specified, only results from these domains will be returned.
        exclude_domains (List[str], optional): List of domains to exclude from search results.
            Cannot be used together with include_domains.
        subpages (int, optional): Number of subpages to retrieve per result. If None, subpages are not retrieved.
        
    Returns:
        List[dict]: List of search responses from Exa API, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the search result
                        'url': str,              # URL of the result
                        'content': str,          # Summary/snippet of content
                        'score': float,          # Relevance score
                        'raw_content': str|None  # Full content or None for secondary citations
                    },
                    ...
                ]
            }
    """
    # Check that include_domains and exclude_domains are not both specified
    if include_domains and exclude_domains:
        raise ValueError("Cannot specify both include_domains and exclude_domains")
    
    # Initialize Exa client (API key should be configured in your .env file)
    exa = Exa(api_key = f"{os.getenv('EXA_API_KEY')}")
    
    # Define the function to process a single query
    async def process_query(query):
        # Use run_in_executor to make the synchronous exa call in a non-blocking way
        loop = asyncio.get_event_loop()
        
        # Define the function for the executor with all parameters
        def exa_search_fn():
            # Build parameters dictionary
            kwargs = {
                # Set text to True if max_characters is None, otherwise use an object with max_characters
                "text": True if max_characters is None else {"max_characters": max_characters},
                "summary": True,  # This is an amazing feature by EXA. It provides an AI generated summary of the content based on the query
                "num_results": num_results
            }
            
            # Add optional parameters only if they are provided
            if subpages is not None:
                kwargs["subpages"] = subpages
                
            if include_domains:
                kwargs["include_domains"] = include_domains
            elif exclude_domains:
                kwargs["exclude_domains"] = exclude_domains
                
            return exa.search_and_contents(query, **kwargs)
        
        response = await loop.run_in_executor(None, exa_search_fn)
        
        # Format the response to match the expected output structure
        formatted_results = []
        seen_urls = set()  # Track URLs to avoid duplicates
        
        # Helper function to safely get value regardless of if item is dict or object
        def get_value(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            else:
                return getattr(item, key, default) if hasattr(item, key) else default
        
        # Access the results from the SearchResponse object
        results_list = get_value(response, 'results', [])
        
        # First process all main results
        for result in results_list:
            # Get the score with a default of 0.0 if it's None or not present
            score = get_value(result, 'score', 0.0)
            
            # Combine summary and text for content if both are available
            text_content = get_value(result, 'text', '')
            summary_content = get_value(result, 'summary', '')
            
            content = text_content
            if summary_content:
                if content:
                    content = f"{summary_content}\n\n{content}"
                else:
                    content = summary_content
            
            title = get_value(result, 'title', '')
            url = get_value(result, 'url', '')
            
            # Skip if we've seen this URL before (removes duplicate entries)
            if url in seen_urls:
                continue
                
            seen_urls.add(url)
            
            # Main result entry
            result_entry = {
                "title": title,
                "url": url,
                "content": content,
                "score": score,
                "raw_content": text_content
            }
            
            # Add the main result to the formatted results
            formatted_results.append(result_entry)
        
        # Now process subpages only if the subpages parameter was provided
        if subpages is not None:
            for result in results_list:
                subpages_list = get_value(result, 'subpages', [])
                for subpage in subpages_list:
                    # Get subpage score
                    subpage_score = get_value(subpage, 'score', 0.0)
                    
                    # Combine summary and text for subpage content
                    subpage_text = get_value(subpage, 'text', '')
                    subpage_summary = get_value(subpage, 'summary', '')
                    
                    subpage_content = subpage_text
                    if subpage_summary:
                        if subpage_content:
                            subpage_content = f"{subpage_summary}\n\n{subpage_content}"
                        else:
                            subpage_content = subpage_summary
                    
                    subpage_url = get_value(subpage, 'url', '')
                    
                    # Skip if we've seen this URL before
                    if subpage_url in seen_urls:
                        continue
                        
                    seen_urls.add(subpage_url)
                    
                    formatted_results.append({
                        "title": get_value(subpage, 'title', ''),
                        "url": subpage_url,
                        "content": subpage_content,
                        "score": subpage_score,
                        "raw_content": subpage_text
                    })
        
        # Collect images if available (only from main results to avoid duplication)
        images = []
        for result in results_list:
            image = get_value(result, 'image')
            if image and image not in images:  # Avoid duplicate images
                images.append(image)
                
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": None,
            "images": images,
            "results": formatted_results
        }
    
    # Process all queries sequentially with delay to respect rate limit
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (0.25s = 4 requests per second, well within the 5/s limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(0.25)
            
            result = await process_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing query '{query}': {str(e)}")
            # Add a placeholder result for failed queries to maintain index alignment
            search_docs.append({
                "query": query,
                "follow_up_questions": None,
                "answer": None,
                "images": [],
                "results": [],
                "error": str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e):
                print("Rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(1.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
async def arxiv_search_async(search_queries, load_max_docs=5, get_full_documents=True, load_all_available_meta=True):
    """
    Performs concurrent searches on arXiv using the ArxivRetriever.

    Args:
        search_queries (List[str]): List of search queries or article IDs
        load_max_docs (int, optional): Maximum number of documents to return per query. Default is 5.
        get_full_documents (bool, optional): Whether to fetch full text of documents. Default is True.
        load_all_available_meta (bool, optional): Whether to load all available metadata. Default is True.

    Returns:
        List[dict]: List of search responses from arXiv, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL (Entry ID) of the paper
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str|None  # Full paper content if available
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # Create retriever for each query
            retriever = ArxivRetriever(
                load_max_docs=load_max_docs,
                get_full_documents=get_full_documents,
                load_all_available_meta=load_all_available_meta
            )
            
            # Run the synchronous retriever in a thread pool
            loop = asyncio.get_event_loop()
            docs = await loop.run_in_executor(None, lambda: retriever.invoke(query))
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Extract metadata
                metadata = doc.metadata
                
                # Use entry_id as the URL (this is the actual arxiv link)
                url = metadata.get('entry_id', '')
                
                # Format content with all useful metadata
                content_parts = []

                # Primary information
                if 'Summary' in metadata:
                    content_parts.append(f"Summary: {metadata['Summary']}")

                if 'Authors' in metadata:
                    content_parts.append(f"Authors: {metadata['Authors']}")

                # Add publication information
                published = metadata.get('Published')
                published_str = published.isoformat() if hasattr(published, 'isoformat') else str(published) if published else ''
                if published_str:
                    content_parts.append(f"Published: {published_str}")

                # Add additional metadata if available
                if 'primary_category' in metadata:
                    content_parts.append(f"Primary Category: {metadata['primary_category']}")

                if 'categories' in metadata and metadata['categories']:
                    content_parts.append(f"Categories: {', '.join(metadata['categories'])}")

                if 'comment' in metadata and metadata['comment']:
                    content_parts.append(f"Comment: {metadata['comment']}")

                if 'journal_ref' in metadata and metadata['journal_ref']:
                    content_parts.append(f"Journal Reference: {metadata['journal_ref']}")

                if 'doi' in metadata and metadata['doi']:
                    content_parts.append(f"DOI: {metadata['doi']}")

                # Get PDF link if available in the links
                pdf_link = ""
                if 'links' in metadata and metadata['links']:
                    for link in metadata['links']:
                        if 'pdf' in link:
                            pdf_link = link
                            content_parts.append(f"PDF: {pdf_link}")
                            break

                # Join all content parts with newlines 
                content = "\n".join(content_parts)
                
                result = {
                    'title': metadata.get('Title', ''),
                    'url': url,  # Using entry_id as the URL
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.page_content if get_full_documents else None
                }
                results.append(result)
                
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process queries sequentially with delay to respect arXiv rate limit (1 request per 3 seconds)
    search_docs = []
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests (3 seconds per ArXiv's rate limit)
            if i > 0:  # Don't delay the first request
                await asyncio.sleep(3.0)
            
            result = await process_single_query(query)
            search_docs.append(result)
        except Exception as e:
            # Handle exceptions gracefully
            print(f"Error processing arXiv query '{query}': {str(e)}")
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # Add additional delay if we hit a rate limit error
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("ArXiv rate limit exceeded. Adding additional delay...")
                await asyncio.sleep(5.0)  # Add a longer delay if we hit a rate limit
    
    return search_docs

@traceable
async def pubmed_search_async(search_queries, top_k_results=5, email=None, api_key=None, doc_content_chars_max=4000):
    """
    Performs concurrent searches on PubMed using the PubMedAPIWrapper.

    Args:
        search_queries (List[str]): List of search queries
        top_k_results (int, optional): Maximum number of documents to return per query. Default is 5.
        email (str, optional): Email address for PubMed API. Required by NCBI.
        api_key (str, optional): API key for PubMed API for higher rate limits.
        doc_content_chars_max (int, optional): Maximum characters for document content. Default is 4000.

    Returns:
        List[dict]: List of search responses from PubMed, one per query. Each response has format:
            {
                'query': str,                    # The original search query
                'follow_up_questions': None,      
                'answer': None,
                'images': [],
                'results': [                     # List of search results
                    {
                        'title': str,            # Title of the paper
                        'url': str,              # URL to the paper on PubMed
                        'content': str,          # Formatted summary with metadata
                        'score': float,          # Relevance score (approximated)
                        'raw_content': str       # Full abstract content
                    },
                    ...
                ]
            }
    """
    
    async def process_single_query(query):
        try:
            # print(f"Processing PubMed query: '{query}'")
            
            # Create PubMed wrapper for the query
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email if email else "your_email@example.com",
                api_key=api_key if api_key else ""
            )
            
            # Run the synchronous wrapper in a thread pool
            loop = asyncio.get_event_loop()
            
            # Use wrapper.lazy_load instead of load to get better visibility
            docs = await loop.run_in_executor(None, lambda: list(wrapper.lazy_load(query)))
            
            print(f"Query '{query}' returned {len(docs)} results")
            
            results = []
            # Assign decreasing scores based on the order
            base_score = 1.0
            score_decrement = 1.0 / (len(docs) + 1) if docs else 0
            
            for i, doc in enumerate(docs):
                # Format content with metadata
                content_parts = []
                
                if doc.get('Published'):
                    content_parts.append(f"Published: {doc['Published']}")
                
                if doc.get('Copyright Information'):
                    content_parts.append(f"Copyright Information: {doc['Copyright Information']}")
                
                if doc.get('Summary'):
                    content_parts.append(f"Summary: {doc['Summary']}")
                
                # Generate PubMed URL from the article UID
                uid = doc.get('uid', '')
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                
                # Join all content parts with newlines
                content = "\n".join(content_parts)
                
                result = {
                    'title': doc.get('Title', ''),
                    'url': url,
                    'content': content,
                    'score': base_score - (i * score_decrement),
                    'raw_content': doc.get('Summary', '')
                }
                results.append(result)
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': results
            }
        except Exception as e:
            # Handle exceptions with more detailed information
            error_msg = f"Error processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Process all queries with a reasonable delay between them
    search_docs = []
    
    # Start with a small delay that increases if we encounter rate limiting
    delay = 1.0  # Start with a more conservative delay
    
    for i, query in enumerate(search_queries):
        try:
            # Add delay between requests
            if i > 0:  # Don't delay the first request
                # print(f"Waiting {delay} seconds before next query...")
                await asyncio.sleep(delay)
            
            result = await process_single_query(query)
            search_docs.append(result)
            
            # If query was successful with results, we can slightly reduce delay (but not below minimum)
            if result.get('results') and len(result['results']) > 0:
                delay = max(0.5, delay * 0.9)  # Don't go below 0.5 seconds
            
        except Exception as e:
            # Handle exceptions gracefully
            error_msg = f"Error in main loop processing PubMed query '{query}': {str(e)}"
            print(error_msg)
            
            search_docs.append({
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
            
            # If we hit an exception, increase delay for next query
            delay = min(5.0, delay * 1.5)  # Don't exceed 5 seconds
    
    return search_docs

@traceable
async def linkup_search(search_queries, depth: Optional[str] = "standard"):
    """
    Performs concurrent web searches using the Linkup API.

    Args:
        search_queries (List[SearchQuery]): List of search queries to process
        depth (str, optional): "standard" (default)  or "deep". More details here https://docs.linkup.so/pages/documentation/get-started/concepts

    Returns:
        List[dict]: List of search responses from Linkup API, one per query. Each response has format:
            {
                'results': [            # List of search results
                    {
                        'title': str,   # Title of the search result
                        'url': str,     # URL of the result
                        'content': str, # Summary/snippet of content
                    },
                    ...
                ]
            }
    """
    client = LinkupClient()
    search_tasks = []
    for query in search_queries:
        search_tasks.append(
                client.async_search(
                    query,
                    depth,
                    output_type="searchResults",
                )
            )

    search_results = []
    for response in await asyncio.gather(*search_tasks):
        search_results.append(
            {
                "results": [
                    {"title": result.name, "url": result.url, "content": result.content}
                    for result in response.results
                ],
            }
        )

    return search_results

@traceable
async def google_search_async(search_queries: Union[str, List[str]], max_results: int = 5, include_raw_content: bool = True):
    """
    Performs concurrent web searches using Google.
    Uses Google Custom Search API if environment variables are set, otherwise falls back to web scraping.

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return per query
        include_raw_content (bool): Whether to fetch full page content

    Returns:
        List[dict]: List of search responses from Google, one per query
    """


    # Check for API credentials from environment variables
    api_key = os.environ.get("GOOGLE_API_KEY")
    cx = os.environ.get("GOOGLE_CX")
    use_api = bool(api_key and cx)
    
    # Handle case where search_queries is a single string
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    
    # Define user agent generator
    def get_useragent():
        """Generates a random user agent string."""
        lynx_version = f"Lynx/{random.randint(2, 3)}.{random.randint(8, 9)}.{random.randint(0, 2)}"
        libwww_version = f"libwww-FM/{random.randint(2, 3)}.{random.randint(13, 15)}"
        ssl_mm_version = f"SSL-MM/{random.randint(1, 2)}.{random.randint(3, 5)}"
        openssl_version = f"OpenSSL/{random.randint(1, 3)}.{random.randint(0, 4)}.{random.randint(0, 9)}"
        return f"{lynx_version} {libwww_version} {ssl_mm_version} {openssl_version}"
    
    # Create executor for running synchronous operations
    executor = None if use_api else concurrent.futures.ThreadPoolExecutor(max_workers=5)
    
    # Use a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(5 if use_api else 2)
    
    async def search_single_query(query):
        async with semaphore:
            try:
                results = []
                
                # API-based search
                if use_api:
                    # The API returns up to 10 results per request
                    for start_index in range(1, max_results + 1, 10):
                        # Calculate how many results to request in this batch
                        num = min(10, max_results - (start_index - 1))
                        
                        # Make request to Google Custom Search API
                        params = {
                            'q': query,
                            'key': api_key,
                            'cx': cx,
                            'start': start_index,
                            'num': num
                        }
                        print(f"Requesting {num} results for '{query}' from Google API...")

                        async with aiohttp.ClientSession() as session:
                            async with session.get('https://www.googleapis.com/customsearch/v1', params=params) as response:
                                if response.status != 200:
                                    error_text = await response.text()
                                    print(f"API error: {response.status}, {error_text}")
                                    break
                                    
                                data = await response.json()
                                
                                # Process search results
                                for item in data.get('items', []):
                                    result = {
                                        "title": item.get('title', ''),
                                        "url": item.get('link', ''),
                                        "content": item.get('snippet', ''),
                                        "score": None,
                                        "raw_content": item.get('snippet', '')
                                    }
                                    results.append(result)
                        
                        # Respect API quota with a small delay
                        await asyncio.sleep(0.2)
                        
                        # If we didn't get a full page of results, no need to request more
                        if not data.get('items') or len(data.get('items', [])) < num:
                            break
                
                # Web scraping based search
                else:
                    # Add delay between requests
                    await asyncio.sleep(0.5 + random.random() * 1.5)
                    print(f"Scraping Google for '{query}'...")

                    # Define scraping function
                    def google_search(query, max_results):
                        try:
                            lang = "en"
                            safe = "active"
                            start = 0
                            fetched_results = 0
                            fetched_links = set()
                            search_results = []
                            
                            while fetched_results < max_results:
                                # Send request to Google
                                resp = requests.get(
                                    url="https://www.google.com/search",
                                    headers={
                                        "User-Agent": get_useragent(),
                                        "Accept": "*/*"
                                    },
                                    params={
                                        "q": query,
                                        "num": max_results + 2,
                                        "hl": lang,
                                        "start": start,
                                        "safe": safe,
                                    },
                                    cookies = {
                                        'CONSENT': 'PENDING+987',  # Bypasses the consent page
                                        'SOCS': 'CAESHAgBEhIaAB',
                                    }
                                )
                                resp.raise_for_status()
                                
                                # Parse results
                                soup = BeautifulSoup(resp.text, "html.parser")
                                result_block = soup.find_all("div", class_="ezO2md")
                                new_results = 0
                                
                                for result in result_block:
                                    link_tag = result.find("a", href=True)
                                    title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                                    description_tag = result.find("span", class_="FrIlee")
                                    
                                    if link_tag and title_tag and description_tag:
                                        link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                                        
                                        if link in fetched_links:
                                            continue
                                        
                                        fetched_links.add(link)
                                        title = title_tag.text
                                        description = description_tag.text
                                        
                                        # Store result in the same format as the API results
                                        search_results.append({
                                            "title": title,
                                            "url": link,
                                            "content": description,
                                            "score": None,
                                            "raw_content": description
                                        })
                                        
                                        fetched_results += 1
                                        new_results += 1
                                        
                                        if fetched_results >= max_results:
                                            break
                                
                                if new_results == 0:
                                    break
                                    
                                start += 10
                                time.sleep(1)  # Delay between pages
                            
                            return search_results
                                
                        except Exception as e:
                            print(f"Error in Google search for '{query}': {str(e)}")
                            return []
                    
                    # Execute search in thread pool
                    loop = asyncio.get_running_loop()
                    search_results = await loop.run_in_executor(
                        executor, 
                        lambda: google_search(query, max_results)
                    )
                    
                    # Process the results
                    results = search_results
                
                # If requested, fetch full page content asynchronously (for both API and web scraping)
                if include_raw_content and results:
                    content_semaphore = asyncio.Semaphore(3)
                    
                    async with aiohttp.ClientSession() as session:
                        fetch_tasks = []
                        
                        async def fetch_full_content(result):
                            async with content_semaphore:
                                url = result['url']
                                headers = {
                                    'User-Agent': get_useragent(),
                                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                                }
                                
                                try:
                                    await asyncio.sleep(0.2 + random.random() * 0.6)
                                    async with session.get(url, headers=headers, timeout=10) as response:
                                        if response.status == 200:
                                            # Check content type to handle binary files
                                            content_type = response.headers.get('Content-Type', '').lower()
                                            
                                            # Handle PDFs and other binary files
                                            if 'application/pdf' in content_type or 'application/octet-stream' in content_type:
                                                # For PDFs, indicate that content is binary and not parsed
                                                result['raw_content'] = f"[Binary content: {content_type}. Content extraction not supported for this file type.]"
                                            else:
                                                try:
                                                    # Try to decode as UTF-8 with replacements for non-UTF8 characters
                                                    html = await response.text(errors='replace')
                                                    soup = BeautifulSoup(html, 'html.parser')
                                                    result['raw_content'] = soup.get_text()
                                                except UnicodeDecodeError as ude:
                                                    # Fallback if we still have decoding issues
                                                    result['raw_content'] = f"[Could not decode content: {str(ude)}]"
                                except Exception as e:
                                    print(f"Warning: Failed to fetch content for {url}: {str(e)}")
                                    result['raw_content'] = f"[Error fetching content: {str(e)}]"
                                return result
                        
                        for result in results:
                            fetch_tasks.append(fetch_full_content(result))
                        
                        updated_results = await asyncio.gather(*fetch_tasks)
                        results = updated_results
                        print(f"Fetched full content for {len(results)} results")
                
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": results
                }
            except Exception as e:
                print(f"Error in Google search for query '{query}': {str(e)}")
                return {
                    "query": query,
                    "follow_up_questions": None,
                    "answer": None,
                    "images": [],
                    "results": []
                }
    
    try:
        # Create tasks for all search queries
        search_tasks = [search_single_query(query) for query in search_queries]
        
        # Execute all searches concurrently
        search_results = await asyncio.gather(*search_tasks)
        
        return search_results
    finally:
        # Only shut down executor if it was created
        if executor:
            executor.shutdown(wait=False)

async def scrape_pages(titles: List[str], urls: List[str]) -> str:
    """
    Scrapes content from a list of URLs and formats it into a readable markdown document.
    
    This function:
    1. Takes a list of page titles and URLs
    2. Makes asynchronous HTTP requests to each URL
    3. Converts HTML content to markdown
    4. Formats all content with clear source attribution
    
    Args:
        titles (List[str]): A list of page titles corresponding to each URL
        urls (List[str]): A list of URLs to scrape content from
        
    Returns:
        str: A formatted string containing the full content of each page in markdown format,
             with clear section dividers and source attribution
    """
    
    # Create an async HTTP client
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        pages = []
        
        # Fetch each URL and convert to markdown
        for url in urls:
            try:
                # Fetch the content
                response = await client.get(url)
                response.raise_for_status()
                
                # Convert HTML to markdown if successful
                if response.status_code == 200:
                    # Handle different content types
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        # Convert HTML to markdown
                        markdown_content = markdownify(response.text)
                        pages.append(markdown_content)
                    else:
                        # For non-HTML content, just mention the content type
                        pages.append(f"Content type: {content_type} (not converted to markdown)")
                else:
                    pages.append(f"Error: Received status code {response.status_code}")
        
            except Exception as e:
                # Handle any exceptions during fetch
                pages.append(f"Error fetching URL: {str(e)}")
        
        # Create formatted output 
        formatted_output = f"Search results: \n\n"
        
        for i, (title, url, page) in enumerate(zip(titles, urls, pages)):
            formatted_output += f"\n\n--- SOURCE {i+1}: {title} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"FULL CONTENT:\n {page}"
            formatted_output += "\n\n" + "-" * 80 + "\n"
        
    return  formatted_output

@tool
async def duckduckgo_search(search_queries: List[str]):
    """Perform searches using DuckDuckGo with retry logic to handle rate limits
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        List[dict]: List of search results
    """
    
    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()
        
        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # Change query slightly and add delay between retries
                        if retry_count > 0:
                            # Random delay with exponential backoff
                            delay = backoff_factor ** retry_count + random.random()
                            print(f"Retry {retry_count}/{max_retries} for query '{query}' after {delay:.2f}s delay")
                            time.sleep(delay)
                            
                            # Add a random element to the query to bypass caching/rate limits
                            modifiers = ['about', 'info', 'guide', 'overview', 'details', 'explained']
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query
                        
                        # Execute search
                        ddg_results = list(ddgs.text(modified_query, max_results=5))
                        
                        # Format results
                        for i, result in enumerate(ddg_results):
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('href', ''),
                                'content': result.get('body', ''),
                                'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                                'raw_content': result.get('body', '')
                            })
                        
                        # Return successful results
                        return {
                            'query': query,
                            'follow_up_questions': None,
                            'answer': None,
                            'images': [],
                            'results': results
                        }
                except Exception as e:
                    # Store the exception and retry
                    last_exception = e
                    retry_count += 1
                    print(f"DuckDuckGo search error: {str(e)}. Retrying {retry_count}/{max_retries}")
                    
                    # If not a rate limit error, don't retry
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        print(f"Non-rate limit error, stopping retries: {str(e)}")
                        break
            
            # If we reach here, all retries failed
            print(f"All retries failed for query '{query}': {str(last_exception)}")
            # Return empty results but with query info preserved
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(last_exception)
            }
            
        return await loop.run_in_executor(None, perform_search)

    # Process queries with delay between them to reduce rate limiting
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # Add delay between queries (except first one)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # Random delay 2-4 seconds
            await asyncio.sleep(delay)
        
        # Process the query
        result = await process_single_query(query)
        search_docs.append(result)
        
        # Safely extract URLs and titles from results, handling empty result cases
        if result['results'] and len(result['results']) > 0:
            for res in result['results']:
                if 'url' in res and 'title' in res:
                    urls.append(res['url'])
                    titles.append(res['title'])
    
    # If we got any valid URLs, scrape the pages
    if urls:
        return await scrape_pages(titles, urls)
    else:
        # Return a formatted error message if no valid URLs were found
        return "No valid search results found. Please try different search queries or use a different search API."

@tool
async def tavily_search(queries: List[str], max_results: int = 5, topic: str = "general") -> str:
    """
    Fetches results from Tavily search API.
    
    Args:
        queries (List[str]): List of search queries
        
    Returns:
        str: A formatted string of search results
    """
    # Use tavily_search_async with include_raw_content=True to get content directly
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True
    )

    # Format the search results directly using the raw_content already provided
    formatted_output = f"Search results: \n\n"
    
    # Deduplicate results by URL
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    # Format the unique results
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:30000]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
    
    if unique_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."

@traceable
async def direct_url_crawl(urls: List[str], 
                         use_playwright: bool = False, 
                         remove_selectors: Optional[List[str]] = None,
                         wait_until: Optional[Literal["commit", "domcontentloaded", "load", "networkidle"]] = "networkidle",
                         wait_for_selector: Optional[str] = None,
                         extract_selectors: Optional[List[str]] = None) -> List[Dict]:
    """
    Crawls a list of URLs directly using either BeautifulSoup or Playwright.
    
    Args:
        urls (List[str]): List of URLs to crawl
        use_playwright (bool, optional): Whether to use Playwright for JavaScript-rendered pages. Defaults to False.
        remove_selectors (List[str], optional): List of CSS selectors to remove from the page. Only used with Playwright.
        wait_until (str, optional): When to consider navigation successful. Options: "commit", "domcontentloaded", "load", "networkidle".
        wait_for_selector (str, optional): Wait for a specific CSS selector to be available on the page.
        extract_selectors (List[str], optional): List of CSS selectors to specifically extract from the page.
        
    Returns:
        List[Dict]: List of search responses, one per URL. Each response has format:
            {
                'query': str,              # The original URL
                'follow_up_questions': None,      
                'answer': None,
                'images': list,
                'results': [               # List of search results
                    {
                        'title': str,      # Title of the webpage
                        'url': str,        # URL of the page
                        'content': str,    # Summary/snippet of content
                        'score': float,    # Relevance score
                        'raw_content': str # Full page content
                    }
                ]
            }
    """
    crawl_results = []
    
    # Handle Playwright import only if needed (lazy import)
    if use_playwright:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError("Playwright is not installed. Install it with 'pip install playwright' and then run 'playwright install chromium'")
    
    async def process_url_with_playwright(url):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate to the page
                nav_options = {}
                if wait_until:
                    nav_options["wait_until"] = wait_until
                
                await page.goto(url, **nav_options)
                
                # Wait for specific selector if provided
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, state="visible")
                
                # Remove any elements that match the selectors
                if remove_selectors:
                    for selector in remove_selectors:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            await element.evaluate("element => element.remove()")
                
                # Extract specific content if selectors provided
                extracted_content = ""
                if extract_selectors:
                    for selector in extract_selectors:
                        elements = await page.query_selector_all(selector)
                        for element in elements:
                            text = await element.inner_text()
                            extracted_content += text + "\n\n"
                
                # Get page title
                title = await page.title()
                
                # Get full page HTML content
                html_content = await page.content()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                
                # Clean up the HTML
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Convert to text and clean whitespace
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                raw_content = "\n".join(chunk for chunk in chunks if chunk)
                
                # If we extracted specific content, use that instead
                if extracted_content:
                    content = extracted_content
                else:
                    # Create a shorter summary for 'content' field
                    content = raw_content[:1000] + "..." if len(raw_content) > 1000 else raw_content
                
                # Get images if available
                images = []
                img_tags = soup.find_all('img')
                for img in img_tags:
                    src = img.get('src')
                    if src and not src.startswith('data:'):
                        # Convert relative URLs to absolute
                        if not src.startswith(('http://', 'https://')):
                            parsed_url = urlparse(url)
                            base = f"{parsed_url.scheme}://{parsed_url.netloc}"
                            if src.startswith('/'):
                                src = base + src
                            else:
                                src = base + '/' + src
                        images.append(src)
                
                await browser.close()
                
                return {
                    'query': url,
                    'follow_up_questions': None,
                    'answer': None,
                    'images': images[:5],  # Limit to 5 images
                    'results': [
                        {
                            'title': title,
                            'url': url,
                            'content': content,
                            'score': 1.0,  # Default high score for direct URL crawl
                            'raw_content': raw_content
                        }
                    ]
                }
        except Exception as e:
            print(f"Error processing URL with Playwright: {url} - {str(e)}")
            return {
                'query': url,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    async def process_url_with_bs4(url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return {
                            'query': url,
                            'follow_up_questions': None,
                            'answer': None,
                            'images': [],
                            'results': [],
                            'error': f"Error fetching URL: Status code {response.status}"
                        }
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Get page title
                    title = soup.title.text if soup.title else "Untitled Page"
                    
                    # Clean up the HTML
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Extract specific content if selectors provided
                    extracted_content = ""
                    if extract_selectors:
                        for selector in extract_selectors:
                            elements = soup.select(selector)
                            for element in elements:
                                extracted_content += element.get_text() + "\n\n"
                    
                    # Convert to text and clean whitespace
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    raw_content = "\n".join(chunk for chunk in chunks if chunk)
                    
                    # If we extracted specific content, use that instead
                    if extracted_content:
                        content = extracted_content
                    else:
                        # Create a shorter summary for 'content' field
                        content = raw_content[:1000] + "..." if len(raw_content) > 1000 else raw_content
                    
                    # Get images if available
                    images = []
                    img_tags = soup.find_all('img')
                    for img in img_tags:
                        src = img.get('src')
                        if src and not src.startswith('data:'):
                            # Convert relative URLs to absolute
                            if not src.startswith(('http://', 'https://')):
                                parsed_url = urlparse(url)
                                base = f"{parsed_url.scheme}://{parsed_url.netloc}"
                                if src.startswith('/'):
                                    src = base + src
                                else:
                                    src = base + '/' + src
                            images.append(src)
                    
                    return {
                        'query': url,
                        'follow_up_questions': None,
                        'answer': None,
                        'images': images[:5],  # Limit to 5 images
                        'results': [
                            {
                                'title': title,
                                'url': url,
                                'content': content,
                                'score': 1.0,  # Default high score for direct URL crawl
                                'raw_content': raw_content
                            }
                        ]
                    }
        except Exception as e:
            print(f"Error processing URL with BeautifulSoup: {url} - {str(e)}")
            return {
                'query': url,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            }
    
    # Choose the method for each URL
    for url in urls:
        try:
            if use_playwright:
                result = await process_url_with_playwright(url)
            else:
                result = await process_url_with_bs4(url)
            
            crawl_results.append(result)
        except Exception as e:
            print(f"Error crawling URL: {url} - {str(e)}")
            crawl_results.append({
                'query': url,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(e)
            })
    
    return crawl_results

@tool
async def direct_url_tool(urls: List[str], use_playwright: bool = False) -> str:
    """
    Crawls specific URLs and returns their content in a readable format.
    
    Args:
        urls (List[str]): List of URLs to crawl
        use_playwright (bool, optional): Whether to use Playwright for JavaScript-rendered content
        
    Returns:
        str: Formatted content from the URLs
    """
    try:
        results = await direct_url_crawl(urls, use_playwright)
        
        # Format the results for output
        formatted_output = f"URL content extraction results:\n\n"
        
        for i, result in enumerate(results, 1):
            url = result.get('query', 'Unknown URL')
            
            if 'error' in result:
                formatted_output += f"--- URL {i}: {url} ---\n"
                formatted_output += f"Error: {result['error']}\n\n"
                continue
            
            if not result.get('results'):
                formatted_output += f"--- URL {i}: {url} ---\n"
                formatted_output += f"No content extracted\n\n"
                continue
            
            # Get the first (and typically only) result for this URL
            web_result = result['results'][0]
            
            formatted_output += f"--- URL {i}: {url} ---\n"
            formatted_output += f"Title: {web_result.get('title', 'Untitled')}\n\n"
            
            # Add extracted content
            if web_result.get('raw_content'):
                # Limit content length for readability
                content = web_result['raw_content'][:30000]
                formatted_output += f"CONTENT:\n{content}\n\n"
            
            # Add images if available
            if result.get('images'):
                formatted_output += f"IMAGES:\n"
                for img_url in result['images']:
                    formatted_output += f"- {img_url}\n"
                formatted_output += "\n"
            
            formatted_output += "-" * 80 + "\n\n"
        
        return formatted_output
    except Exception as e:
        return f"Error crawling URLs: {str(e)}"

async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> str:
    """Select and execute the appropriate search API.
    
    Args:
        search_api: Name of the search API to use
        query_list: List of search queries to execute
        params_to_pass: Parameters to pass to the search API
        
    Returns:
        Formatted string containing search results
        
    Raises:
        ValueError: If an unsupported search API is specified
    """
    if search_api == "tavily":
        # Tavily search tool used with both workflow and agent 
        return await tavily_search.ainvoke({'queries': query_list}, **params_to_pass)
    elif search_api == "duckduckgo":
        # DuckDuckGo search tool used with both workflow and agent 
        return await duckduckgo_search.ainvoke({'search_queries': query_list})
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "linkup":
        search_results = await linkup_search(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "googlesearch":
        search_results = await google_search_async(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    elif search_api == "direct_url":
        # For direct URL crawling, the query_list contains URLs
        search_results = await direct_url_crawl(query_list, **params_to_pass)
        return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")
