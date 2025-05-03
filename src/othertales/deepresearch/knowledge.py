"""
Knowledge retrieval nodes for checking existing information.

This module provides components that integrate with the research graph to check
existing knowledge before proceeding with web research. It enables the system to
use previously verified information, reducing redundant searches and ensuring
consistency in responses.

Main components:
- KnowledgeRetriever: Core node for retrieving knowledge from the document store
- DocumentProcessor: Processes retrieved documents into structured information
- KnowledgeGraphExplorer: Explores related entities in the knowledge graph
- LegalKnowledgeInterface: Specialized interface for UK legal information
- TaxKnowledgeInterface: Specialized interface for HMRC tax information
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Literal

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Import our components
from othertales.deepresearch.state import (
    Source, KnowledgeResult, AgentMode, SectionState, SearchQuery
)
from othertales.deepresearch.storage import (
    DocumentStore, VerifiedDocument, VerificationStatus, create_document_store
)
from othertales.deepresearch.configuration import Configuration
from othertales.deepresearch.utils import get_config_value

# Configure logging
import logging
logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Retrieves relevant knowledge from document store."""
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        similarity_threshold: float = 0.75,
        max_results: int = 5,
        config: Optional[RunnableConfig] = None,
    ):
        """Initialize the knowledge retriever.
        
        Args:
            document_store: Document store to retrieve from.
            similarity_threshold: Minimum similarity score for retrieval.
            max_results: Maximum number of results to return.
            config: Optional configuration.
        """
        # Initialize document store
        if document_store:
            self.document_store = document_store
        else:
            # Try to create document store from environment variables
            try:
                self.document_store = create_document_store()
            except Exception as e:
                logger.warning(f"Failed to create document store: {e}")
                self.document_store = None
                
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Initialize LLM for relevance assessment
        self._init_llm(config)
        
    def _init_llm(self, config: Optional[RunnableConfig] = None):
        """Initialize LLM for relevance assessment."""
        # Get configuration
        configurable = None
        if config:
            configurable = Configuration.from_runnable_config(config)
        
        # Default model settings
        model_provider = "anthropic"
        model_name = "claude-3-5-sonnet-latest"
        model_kwargs = {}
        
        # Use configuration if available
        if configurable:
            model_provider = get_config_value(configurable.writer_provider or model_provider)
            model_name = get_config_value(configurable.writer_model or model_name)
            model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
        
        # Initialize LLM
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            model_kwargs=model_kwargs
        )
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeResult)
        
    async def retrieve_knowledge(
        self,
        query: str,
        mode: AgentMode = "general",
        k: int = None,
    ) -> KnowledgeResult:
        """Retrieve knowledge from the document store.
        
        Args:
            query: Query to search for.
            mode: Mode of operation (legal, tax, general).
            k: Number of results to return. If None, uses self.max_results.
            
        Returns:
            Knowledge result with retrieved information.
        """
        if not self.document_store:
            # Return empty result if no document store
            return KnowledgeResult(
                has_relevant_information=False,
                sources=[],
                relevance_score=0.0,
                content_summary="No document store available for knowledge retrieval."
            )
        
        # Use provided k or default
        k = k or self.max_results
        
        try:
            # Perform similarity search
            similar_docs = self.document_store.similarity_search(
                query=query,
                k=k,
                domain=mode,
                verification_status=VerificationStatus.VERIFIED
            )
            
            if not similar_docs:
                # Return empty result if no documents found
                return KnowledgeResult(
                    has_relevant_information=False,
                    sources=[],
                    relevance_score=0.0,
                    content_summary="No relevant information found in knowledge base."
                )
            
            # Convert to Source objects
            sources = [
                Source(
                    id=doc.id,
                    title=doc.title,
                    url=doc.url,
                    content=doc.content,
                    domain=doc.domain,
                    verification_status=doc.verification_status,
                    metadata=doc.metadata
                )
                for doc in similar_docs
            ]
            
            # Assess relevance of retrieved documents
            relevance_assessment = await self._assess_relevance(query, sources)
            
            return relevance_assessment
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}")
            # Return empty result on error
            return KnowledgeResult(
                has_relevant_information=False,
                sources=[],
                relevance_score=0.0,
                content_summary=f"Error retrieving knowledge: {str(e)}"
            )
    
    async def _assess_relevance(
        self,
        query: str,
        sources: List[Source]
    ) -> KnowledgeResult:
        """Assess the relevance of retrieved sources to the query.
        
        Args:
            query: Original query.
            sources: Retrieved sources.
            
        Returns:
            Knowledge result with relevance assessment.
        """
        # Create prompt for relevance assessment
        system_prompt = f"""You are a Knowledge Assessment Agent responsible for evaluating the relevance of information to a specific query.

Your task is to evaluate the sources provided and determine:
1. Whether they contain information relevant to the query
2. How relevant the information is (score from 0.0 to 1.0)
3. A concise summary of the relevant information

The output must be a valid JSON object matching this schema:
{self.parser.get_format_instructions()}

Guidelines for relevance assessment:
- Consider exactness of match, comprehensiveness, and currency of information
- For legal content, prioritize relevance to specific legal questions, jurisdictions, and applicable laws
- For tax content, prioritize relevance to specific tax questions, applicable rules, and current tax years
- Be critical - only mark information as relevant if it genuinely addresses the query
"""

        # Format query and sources for the prompt
        query_content = f"QUERY: {query}\n\n"
        
        sources_content = "SOURCES:\n\n"
        for i, source in enumerate(sources, 1):
            sources_content += f"""SOURCE {i}:
Title: {source.title}
URL: {source.url or 'Not specified'}
Domain: {source.domain}
Content:
{source.content[:1000]}... [truncated if longer]

"""

        # Create assessment instructions
        assessment_instructions = """
Please assess these sources and determine their relevance to the query.
Your output must include:
1. Whether relevant information was found (true/false)
2. The list of sources (maintain their order and all metadata)
3. A relevance score between 0.0 and 1.0
4. A concise summary of the relevant information (or explanation of why no relevant information was found)
"""

        try:
            # Invoke the LLM
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query_content + sources_content + assessment_instructions)
            ])
            
            # Parse the result
            try:
                # Try to extract JSON from markdown code blocks if present
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Parse the result
                result = self.parser.parse(content)
                
                # Ensure sources are preserved
                if not result.sources:
                    result.sources = sources
                
                return result
            
            except Exception as e:
                logger.error(f"Failed to parse relevance assessment: {e}")
                # Return basic result with the sources
                return KnowledgeResult(
                    has_relevant_information=len(sources) > 0,
                    sources=sources,
                    relevance_score=0.5 if sources else 0.0,
                    content_summary=f"Relevance assessment failed: {str(e)}. Using sources as-is."
                )
                
        except Exception as e:
            logger.error(f"Error assessing relevance: {e}")
            # Return basic result with the sources
            return KnowledgeResult(
                has_relevant_information=len(sources) > 0,
                sources=sources,
                relevance_score=0.5 if sources else 0.0,
                content_summary=f"Error assessing relevance: {str(e)}. Using sources as-is."
            )


class LegalKnowledgeInterface:
    """Specialized interface for UK legal knowledge."""
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        config: Optional[RunnableConfig] = None,
    ):
        """Initialize the legal knowledge interface.
        
        Args:
            document_store: Document store to retrieve from.
            config: Optional configuration.
        """
        # Initialize base retriever with legal focus
        self.retriever = KnowledgeRetriever(
            document_store=document_store,
            config=config
        )
        
        # Initialize LLM for legal analysis
        self._init_llm(config)
        
    def _init_llm(self, config: Optional[RunnableConfig] = None):
        """Initialize LLM for legal analysis."""
        # Get configuration
        configurable = None
        if config:
            configurable = Configuration.from_runnable_config(config)
        
        # Default model settings
        model_provider = "anthropic"
        model_name = "claude-3-sonnet-latest"
        model_kwargs = {}
        
        # Use configuration if available
        if configurable:
            model_provider = get_config_value(configurable.legal_provider or model_provider)
            model_name = get_config_value(configurable.legal_model or model_name)
            model_kwargs = get_config_value(configurable.legal_model_kwargs or {})
        
        # Initialize LLM
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            model_kwargs=model_kwargs
        )
    
    async def retrieve_legal_knowledge(
        self,
        query: str,
        jurisdiction: Literal["UK", "England", "Wales", "Scotland", "Northern Ireland"] = "UK",
        legal_area: Optional[str] = None,
        k: int = 5,
    ) -> KnowledgeResult:
        """Retrieve legal knowledge with enhanced legal context.
        
        Args:
            query: Legal query to search for.
            jurisdiction: Specific UK jurisdiction.
            legal_area: Specific area of law.
            k: Number of results to return.
            
        Returns:
            Knowledge result with retrieved legal information.
        """
        # Enhance query with legal context
        enhanced_query = self._enhance_legal_query(query, jurisdiction, legal_area)
        
        # Retrieve knowledge with legal mode
        base_result = await self.retriever.retrieve_knowledge(
            query=enhanced_query,
            mode="legal",
            k=k
        )
        
        # If no relevant information, return as is
        if not base_result.has_relevant_information or not base_result.sources:
            return base_result
        
        # Enhance retrieval with legal analysis
        return await self._enhance_legal_analysis(query, base_result, jurisdiction, legal_area)
    
    def _enhance_legal_query(
        self,
        query: str,
        jurisdiction: str,
        legal_area: Optional[str]
    ) -> str:
        """Enhance query with legal context for better retrieval.
        
        Args:
            query: Original query.
            jurisdiction: Specific UK jurisdiction.
            legal_area: Specific area of law.
            
        Returns:
            Enhanced query.
        """
        # Start with the original query
        enhanced = query
        
        # Add jurisdiction context if not already in query
        if jurisdiction and jurisdiction.lower() not in query.lower():
            enhanced += f" in {jurisdiction} jurisdiction"
            
        # Add legal area context if provided and not already in query
        if legal_area and legal_area.lower() not in query.lower():
            enhanced += f" regarding {legal_area} law"
            
        return enhanced
    
    async def _enhance_legal_analysis(
        self,
        original_query: str,
        base_result: KnowledgeResult,
        jurisdiction: str,
        legal_area: Optional[str]
    ) -> KnowledgeResult:
        """Enhance retrieval with specialized legal analysis.
        
        Args:
            original_query: Original query.
            base_result: Base knowledge result.
            jurisdiction: Specific UK jurisdiction.
            legal_area: Specific area of law.
            
        Returns:
            Enhanced knowledge result.
        """
        # Format legal context for the prompt
        legal_context = f"""
JURISDICTION: {jurisdiction}
LEGAL AREA: {legal_area or 'Not specified'}
"""

        # Create prompt for legal analysis
        system_prompt = """You are a UK Legal Knowledge Specialist responsible for analyzing legal information for accuracy and relevance.

Your task is to analyze the legal sources provided in relation to a specific legal query, considering:
1. Jurisdictional applicability
2. Currency of the law (whether it's still in effect)
3. Relevance to the specific legal question
4. Proper interpretation of statutes, case law, and legal principles

For UK legal information, be aware of jurisdictional differences between England & Wales, Scotland, and Northern Ireland.
"""

        # Format query and sources
        query_content = f"LEGAL QUERY: {original_query}\n\n{legal_context}\n\n"
        
        sources_content = "LEGAL SOURCES:\n\n"
        for i, source in enumerate(base_result.sources, 1):
            sources_content += f"""SOURCE {i}:
Title: {source.title}
URL: {source.url or 'Not specified'}
Content:
{source.content[:1000]}... [truncated if longer]

"""

        # Create analysis instructions
        analysis_instructions = """
Please analyze these legal sources and provide:
1. An assessment of their accuracy and relevance to the legal query
2. Any jurisdictional considerations that should be noted
3. Whether the legal information is current and applicable
4. A concise summary of the legal position based on these sources

Focus on providing legally sound analysis that would be helpful for addressing the query within the specified jurisdiction.
"""

        try:
            # Invoke the LLM for legal analysis
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query_content + sources_content + analysis_instructions)
            ])
            
            # Update the content summary with legal analysis
            legal_analysis = response.content
            
            # Create enhanced result
            return KnowledgeResult(
                has_relevant_information=base_result.has_relevant_information,
                sources=base_result.sources,
                relevance_score=base_result.relevance_score,
                content_summary=f"""LEGAL ANALYSIS:
{legal_analysis}

ORIGINAL SUMMARY:
{base_result.content_summary}"""
            )
            
        except Exception as e:
            logger.error(f"Error enhancing legal analysis: {e}")
            # Return original result on error
            return base_result


class TaxKnowledgeInterface:
    """Specialized interface for HMRC tax knowledge."""
    
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        config: Optional[RunnableConfig] = None,
    ):
        """Initialize the tax knowledge interface.
        
        Args:
            document_store: Document store to retrieve from.
            config: Optional configuration.
        """
        # Initialize base retriever with tax focus
        self.retriever = KnowledgeRetriever(
            document_store=document_store,
            config=config
        )
        
        # Initialize LLM for tax analysis
        self._init_llm(config)
        
    def _init_llm(self, config: Optional[RunnableConfig] = None):
        """Initialize LLM for tax analysis."""
        # Get configuration
        configurable = None
        if config:
            configurable = Configuration.from_runnable_config(config)
        
        # Default model settings
        model_provider = "anthropic"
        model_name = "claude-3-sonnet-latest"
        model_kwargs = {}
        
        # Use configuration if available
        if configurable:
            model_provider = get_config_value(configurable.tax_provider or model_provider)
            model_name = get_config_value(configurable.tax_model or model_name)
            model_kwargs = get_config_value(configurable.tax_model_kwargs or {})
        
        # Initialize LLM
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            model_kwargs=model_kwargs
        )
    
    async def retrieve_tax_knowledge(
        self,
        query: str,
        tax_year: Optional[str] = None,
        tax_type: Optional[str] = None,
        k: int = 5,
    ) -> KnowledgeResult:
        """Retrieve tax knowledge with enhanced tax context.
        
        Args:
            query: Tax query to search for.
            tax_year: Specific tax year.
            tax_type: Specific type of tax.
            k: Number of results to return.
            
        Returns:
            Knowledge result with retrieved tax information.
        """
        # Enhance query with tax context
        enhanced_query = self._enhance_tax_query(query, tax_year, tax_type)
        
        # Retrieve knowledge with tax mode
        base_result = await self.retriever.retrieve_knowledge(
            query=enhanced_query,
            mode="tax",
            k=k
        )
        
        # If no relevant information, return as is
        if not base_result.has_relevant_information or not base_result.sources:
            return base_result
        
        # Enhance retrieval with tax analysis
        return await self._enhance_tax_analysis(query, base_result, tax_year, tax_type)
    
    def _enhance_tax_query(
        self,
        query: str,
        tax_year: Optional[str],
        tax_type: Optional[str]
    ) -> str:
        """Enhance query with tax context for better retrieval.
        
        Args:
            query: Original query.
            tax_year: Specific tax year.
            tax_type: Specific type of tax.
            
        Returns:
            Enhanced query.
        """
        # Start with the original query
        enhanced = query
        
        # Add tax year context if provided and not already in query
        if tax_year and tax_year.lower() not in query.lower():
            enhanced += f" for tax year {tax_year}"
            
        # Add tax type context if provided and not already in query
        if tax_type and tax_type.lower() not in query.lower():
            enhanced += f" regarding {tax_type}"
            
        return enhanced
    
    async def _enhance_tax_analysis(
        self,
        original_query: str,
        base_result: KnowledgeResult,
        tax_year: Optional[str],
        tax_type: Optional[str]
    ) -> KnowledgeResult:
        """Enhance retrieval with specialized tax analysis.
        
        Args:
            original_query: Original query.
            base_result: Base knowledge result.
            tax_year: Specific tax year.
            tax_type: Specific type of tax.
            
        Returns:
            Enhanced knowledge result.
        """
        # Format tax context for the prompt
        tax_context = f"""
TAX YEAR: {tax_year or 'Current tax year'}
TAX TYPE: {tax_type or 'Not specified'}
"""

        # Create prompt for tax analysis
        system_prompt = """You are an HMRC Tax Knowledge Specialist responsible for analyzing tax information for accuracy and relevance.

Your task is to analyze the tax sources provided in relation to a specific tax query, considering:
1. Current tax year applicability
2. Correct tax rates and thresholds
3. Relevant tax legislation and HMRC guidance
4. Tax optimization opportunities and compliance requirements

Be precise about tax figures, calculations, and filing requirements. Always clarify if information applies to specific tax years.
"""

        # Format query and sources
        query_content = f"TAX QUERY: {original_query}\n\n{tax_context}\n\n"
        
        sources_content = "TAX SOURCES:\n\n"
        for i, source in enumerate(base_result.sources, 1):
            sources_content += f"""SOURCE {i}:
Title: {source.title}
URL: {source.url or 'Not specified'}
Content:
{source.content[:1000]}... [truncated if longer]

"""

        # Create analysis instructions
        analysis_instructions = """
Please analyze these tax sources and provide:
1. An assessment of their accuracy and relevance to the tax query
2. Whether the tax information is current for the specified tax year
3. Any considerations for optimizing tax position while maintaining compliance
4. A concise summary of the tax position based on these sources

Focus on providing accurate tax guidance that would help address the query while ensuring HMRC compliance.
"""

        try:
            # Invoke the LLM for tax analysis
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query_content + sources_content + analysis_instructions)
            ])
            
            # Update the content summary with tax analysis
            tax_analysis = response.content
            
            # Create enhanced result
            return KnowledgeResult(
                has_relevant_information=base_result.has_relevant_information,
                sources=base_result.sources,
                relevance_score=base_result.relevance_score,
                content_summary=f"""TAX ANALYSIS:
{tax_analysis}

ORIGINAL SUMMARY:
{base_result.content_summary}"""
            )
            
        except Exception as e:
            logger.error(f"Error enhancing tax analysis: {e}")
            # Return original result on error
            return base_result


# Graph node function for knowledge retrieval
async def check_existing_knowledge(state: SectionState, config: RunnableConfig):
    """Node function that checks for existing knowledge before web search.
    
    Args:
        state: Current section state.
        config: Configuration.
        
    Returns:
        Dict with knowledge retrieval results.
    """
    # Get state
    topic = state["topic"]
    section = state["section"]
    mode = state.get("mode", "general")
    search_queries = state.get("search_queries", [])
    
    # Initialize document store
    try:
        document_store = create_document_store()
    except Exception as e:
        logger.warning(f"Failed to create document store: {e}")
        return {
            "knowledge_results": KnowledgeResult(
                has_relevant_information=False,
                sources=[],
                relevance_score=0.0,
                content_summary=f"Failed to create document store: {str(e)}"
            )
        }
    
    # Create query from section topic and description
    main_query = f"{topic}: {section.description}"
    
    try:
        # Select appropriate interface based on mode
        if mode == "legal":
            interface = LegalKnowledgeInterface(document_store, config)
            result = await interface.retrieve_legal_knowledge(main_query)
        elif mode == "tax":
            interface = TaxKnowledgeInterface(document_store, config)
            result = await interface.retrieve_tax_knowledge(main_query)
        else:
            # General mode
            retriever = KnowledgeRetriever(document_store, config=config)
            result = await retriever.retrieve_knowledge(main_query, mode=mode)
        
        # If we have search queries, also check those
        if search_queries and not result.has_relevant_information:
            retriever = KnowledgeRetriever(document_store, config=config)
            for query in search_queries:
                # Skip empty queries
                if not query.search_query:
                    continue
                    
                # Run retrieval for this query
                query_result = await retriever.retrieve_knowledge(
                    query.search_query, mode=mode
                )
                
                # If we found something, use this result
                if query_result.has_relevant_information:
                    result = query_result
                    break
        
        return {"knowledge_results": result}
    
    except Exception as e:
        logger.error(f"Error checking existing knowledge: {e}")
        return {
            "knowledge_results": KnowledgeResult(
                has_relevant_information=False,
                sources=[],
                relevance_score=0.0,
                content_summary=f"Error checking existing knowledge: {str(e)}"
            )
        }


# Decision function to check if we need web search
def should_perform_web_search(state: SectionState) -> Literal["search_web", "write_section_from_knowledge"]:
    """Decision function that determines if web search is needed.
    
    Args:
        state: Current section state.
        
    Returns:
        Next node to visit: either "search_web" or "write_section_from_knowledge".
    """
    # Get knowledge results
    knowledge_results = state.get("knowledge_results")
    
    # If no knowledge results or no relevant information, perform web search
    if not knowledge_results or not knowledge_results.has_relevant_information:
        return "search_web"
    
    # If relevance score is below threshold, perform web search
    if knowledge_results.relevance_score < 0.7:  # Threshold can be adjusted
        return "search_web"
    
    # Otherwise, use existing knowledge
    return "write_section_from_knowledge"


# Node function for writing section from knowledge
async def write_section_from_knowledge(state: SectionState, config: RunnableConfig):
    """Node function that writes a section using existing knowledge.
    
    Args:
        state: Current section state.
        config: Configuration.
        
    Returns:
        Dict with updated section and sources.
    """
    # Get state
    topic = state["topic"]
    section = state["section"]
    knowledge_results = state["knowledge_results"]
    
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    
    # Format sources for writing
    sources_str = ""
    for i, source in enumerate(knowledge_results.sources, 1):
        sources_str += f"""SOURCE {i}:
Title: {source.title}
URL: {source.url or 'Not specified'}
Content:
{source.content}

"""
    
    # Format system instructions
    section_writer_instructions = """Write one section of a research report using the provided knowledge sources.

Your task is to:
1. Review the report topic, section name, and section topic carefully.
2. Use ONLY the provided knowledge sources to write the section.
3. Create a well-structured, informative section that directly addresses the topic.
4. Include proper citations to the sources used.

Writing Guidelines:
- Focus on accuracy and completeness
- Use simple, clear language
- Keep paragraphs short (2-3 sentences max)
- Use ## for section title (Markdown format)
- Maximum length of 300 words

Citation Rules:
- Assign each source a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- Number sources sequentially (1,2,3...)
- Example format: [1] Source Title: URL

Quality Check:
- Ensure EVERY claim is supported by the provided sources
- Maintain factual accuracy throughout
"""

    # Format section inputs
    section_inputs = f"""
Report Topic: {topic}

Section Name: {section.name}

Section Topic: {section.description}

Knowledge Summary: {knowledge_results.content_summary}

Knowledge Sources:
{sources_str}
"""

    # Generate section
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    
    section_content = await writer_model.ainvoke([
        SystemMessage(content=section_writer_instructions),
        HumanMessage(content=section_inputs)
    ])
    
    # Write content to the section object
    section.content = section_content.content
    
    # Update section with sources
    section.sources = knowledge_results.sources
    section.verified = True  # Sources from knowledge base are already verified
    
    # Return updated section and sources
    return {
        "section": section,
        "sources": knowledge_results.sources,
        "completed_sections": [section]
    }