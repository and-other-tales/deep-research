import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict
from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic

3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"
    GOOGLESEARCH = "googlesearch"
    DIRECT_URL = "direct_url"

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"
    HUGGINGFACE = "huggingface"
    HUGGINGFACE_ENDPOINT = "huggingface-endpoint"
    GOOGLE = "google"

@dataclass(kw_only=True)
class Configuration:
    # Common
    report_structure: str = os.environ.get("REPORT_STRUCTURE", DEFAULT_REPORT_STRUCTURE)
    search_api: SearchAPI = SearchAPI(os.environ.get("SEARCH_API", "tavily").lower())
    search_api_config: Optional[Dict[str, Any]] = None
    agent_mode: str = os.environ.get("AGENT_MODE", "general")

    # LLM
    llm_provider: str = os.environ.get("LLM_PROVIDER", "anthropic")

    # Graph
    number_of_queries: int = int(os.environ.get("NUMBER_OF_QUERIES", "2"))
    max_search_depth: int = int(os.environ.get("MAX_SEARCH_DEPTH", "2"))
    planner_provider: str = os.environ.get("PLANNER_PROVIDER", os.environ.get("LLM_PROVIDER", "anthropic"))
    planner_model: str = os.environ.get("PLANNER_MODEL", "claude-3-7-sonnet-latest")
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    writer_provider: str = os.environ.get("WRITER_PROVIDER", os.environ.get("LLM_PROVIDER", "anthropic"))
    writer_model: str = os.environ.get("WRITER_MODEL", "claude-3-5-sonnet-latest")
    writer_model_kwargs: Optional[Dict[str, Any]] = None

    # Knowledge/verification
    use_knowledge_storage: bool = os.environ.get("USE_KNOWLEDGE_STORAGE", "true").lower() == "true"
    verification_required: bool = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    min_verification_confidence: float = float(os.environ.get("MIN_VERIFICATION_CONFIDENCE", "0.7"))

    verification_provider: str = os.environ.get("VERIFICATION_PROVIDER", os.environ.get("LLM_PROVIDER", "anthropic"))
    fact_check_model: str = os.environ.get("FACT_CHECK_MODEL", "claude-3-5-sonnet-latest")
    source_eval_model: str = os.environ.get("SOURCE_EVAL_MODEL", "claude-3-5-sonnet-latest")
    consistency_model: str = os.environ.get("CONSISTENCY_MODEL", "claude-3-5-sonnet-latest")
    legal_model: str = os.environ.get("LEGAL_MODEL", "claude-3-sonnet-latest")
    tax_model: str = os.environ.get("TAX_MODEL", "claude-3-sonnet-latest")

    legal_provider: str = os.environ.get("LEGAL_PROVIDER", os.environ.get("LLM_PROVIDER", "anthropic"))
    legal_model_kwargs: Optional[Dict[str, Any]] = None
    tax_provider: str = os.environ.get("TAX_PROVIDER", os.environ.get("LLM_PROVIDER", "anthropic"))
    tax_model_kwargs: Optional[Dict[str, Any]] = None

    # Multi-agent models
    supervisor_model: str = os.environ.get("SUPERVISOR_MODEL", "openai:gpt-4.1")
    researcher_model: str = os.environ.get("RESEARCHER_MODEL", "openai:gpt-4.1")

    # Knowledge DBs
    mongodb_uri: str = os.environ.get("MONGODB_URI", "")
    neo4j_uri: str = os.environ.get("NEO4J_URI", "")
    neo4j_username: str = os.environ.get("NEO4J_USERNAME", "")
    neo4j_password: str = os.environ.get("NEO4J_PASSWORD", "")

    # Server
    server_host: str = os.environ.get("SERVER_HOST", "127.0.0.1")
    server_port: int = int(os.environ.get("SERVER_PORT", "8080"))
    internal_host: str = os.environ.get("INTERNAL_HOST", "127.0.0.1")
    langgraph_port: int = int(os.environ.get("LANGGRAPH_PORT", "8081"))

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        values = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        # Convert types if needed (e.g., booleans, ints)
        if "number_of_queries" in values:
            values["number_of_queries"] = int(values["number_of_queries"])
        if "max_search_depth" in values:
            values["max_search_depth"] = int(values["max_search_depth"])
        if "use_knowledge_storage" in values:
            values["use_knowledge_storage"] = str(values["use_knowledge_storage"]).lower() == "true"
        if "verification_required" in values:
            values["verification_required"] = str(values["verification_required"]).lower() == "true"
        if "min_verification_confidence" in values:
            values["min_verification_confidence"] = float(values["min_verification_confidence"])
        if "search_api" in values:
            values["search_api"] = SearchAPI(values["search_api"].lower())

        return cls(**{k: v for k, v in values.items() if v is not None})
