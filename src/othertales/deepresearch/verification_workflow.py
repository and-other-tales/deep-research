"""
Verification workflow for research and document processing.

This module provides components that integrate with the research graph to
verify information obtained from web searches and other sources. It ensures
that all information used in reports meets strict accuracy and quality standards.

Main components:
- process_sources: Extracts sources from web search results
- verify_sources: Verifies sources using the verification agents
- save_verified_sources: Saves verified sources to document store
- handle_verification_failure: Manages follow-up actions for failed verification
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Import our components
from othertales.deepresearch.state import (
    Source, SectionState, SearchQuery, VerificationFeedback
)
from othertales.deepresearch.storage import (
    DocumentStore, VerifiedDocument, VerificationStatus, create_document_store
)
from othertales.deepresearch.verification import (
    VerificationType, VerificationCoordinator, create_verification_coordinator
)
from othertales.deepresearch.configuration import Configuration
from othertales.deepresearch.utils import get_config_value

# Configure logging
import logging
logger = logging.getLogger(__name__)


async def process_sources(state: SectionState, config: RunnableConfig):
    """Process web search results into structured sources.
    
    Args:
        state: Current section state.
        config: Configuration.
        
    Returns:
        Dict with extracted sources.
    """
    # Get state
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]
    mode = state.get("mode", "general")
    
    # Split source string into individual sources
    sources = []
    current_source = None
    
    # Simple parser for the formatted source string
    lines = source_str.split("\n")
    in_content = False
    
    for line in lines:
        if line.startswith("Source:"):
            # Save previous source if it exists
            if current_source:
                sources.append(current_source)
            
            # Start new source
            current_source = {
                "title": line[7:].strip(),
                "url": "",
                "content": ""
            }
            in_content = False
        elif line.startswith("URL:") and current_source:
            current_source["url"] = line[4:].strip()
            in_content = False
        elif line.startswith("Most relevant content from source:") and current_source:
            in_content = True
            # Skip this line itself
        elif line.startswith("Full source content limited to") and current_source:
            # This line starts the full content section
            # Append to existing content
            in_content = True
        elif line.startswith("===") or line.startswith("====="):
            # Section separator, skip
            continue
        elif current_source and in_content:
            # Add to content
            if current_source["content"]:
                current_source["content"] += "\n"
            current_source["content"] += line
    
    # Add the last source if it exists
    if current_source:
        sources.append(current_source)
    
    # Convert to Source objects
    source_objects = [
        Source(
            title=source["title"],
            url=source["url"],
            content=source["content"],
            domain=mode,
            verification_status=VerificationStatus.PENDING,
            metadata={"section": section.name, "topic": topic}
        )
        for source in sources
        if source["content"].strip()  # Skip empty sources
    ]
    
    return {"sources": source_objects}


async def verify_sources(state: SectionState, config: RunnableConfig):
    """Verify sources using the verification system.
    
    Args:
        state: Current section state.
        config: Configuration.
        
    Returns:
        Dict with verification results.
    """
    # Get state
    sources = state["sources"]
    mode = state.get("mode", "general")
    
    # Skip if no sources
    if not sources:
        return {
            "verification_feedback": VerificationFeedback(
                verification_status=VerificationStatus.PENDING,
                verification_details={},
                verification_issues=["No sources to verify"],
                verification_passed=False,
                follow_up_actions=["Generate new search queries to find sources"]
            )
        }
    
    try:
        # Create document store
        document_store = create_document_store()
        
        # Create verification coordinator
        verification_coordinator = create_verification_coordinator(document_store, config)
        
        # Create verification type based on mode
        verification_type = VerificationType(
            factual_accuracy=True,
            source_reliability=True,
            internal_consistency=True,
            legal_correctness=(mode == "legal"),
            tax_compliance=(mode == "tax")
        )
        
        # Process each source
        verification_results = {}
        verification_issues = []
        all_passed = True
        
        for source in sources:
            # Convert to VerifiedDocument
            doc = VerifiedDocument(
                id=source.id,
                content=source.content,
                title=source.title,
                url=source.url,
                domain=mode,
                metadata=source.metadata
            )
            
            # Verify the document
            passed, results = await verification_coordinator.verify_document(
                doc, verification_type
            )
            
            # Store results
            verification_results[source.id] = results
            
            # Update overall status
            all_passed = all_passed and passed
            
            # Collect issues
            if not passed:
                for agent_name, result in results.items():
                    if result.status == "fail":
                        for issue in result.issues:
                            verification_issues.append(
                                f"Source '{source.title}': {issue} (Agent: {agent_name})"
                            )
        
        # Determine overall verification status
        if all_passed:
            verification_status = VerificationStatus.VERIFIED
            follow_up_actions = []
        else:
            verification_status = VerificationStatus.REJECTED
            follow_up_actions = [
                "Refine search queries to find more reliable sources",
                "Review verification issues and address specific concerns",
                "Consider using alternative search approaches"
            ]
        
        # Create verification feedback
        verification_feedback = VerificationFeedback(
            verification_status=verification_status,
            verification_details=verification_results,
            verification_issues=verification_issues,
            verification_passed=all_passed,
            follow_up_actions=follow_up_actions
        )
        
        return {"verification_feedback": verification_feedback}
    
    except Exception as e:
        logger.error(f"Error verifying sources: {e}")
        # Return error feedback
        return {
            "verification_feedback": VerificationFeedback(
                verification_status=VerificationStatus.PENDING,
                verification_details={"error": str(e)},
                verification_issues=[f"Verification process failed: {str(e)}"],
                verification_passed=False,
                follow_up_actions=["Retry verification with different sources"]
            )
        }


async def save_verified_sources(state: SectionState, config: RunnableConfig):
    """Save verified sources to the document store.
    
    Args:
        state: Current section state.
        config: Configuration.
        
    Returns:
        Dict with updated sources.
    """
    # Get state
    sources = state["sources"]
    verification_feedback = state["verification_feedback"]
    
    # Skip if no sources or verification failed
    if not sources or not verification_feedback.verification_passed:
        return {"sources": sources}
    
    try:
        # Create document store
        document_store = create_document_store()
        
        # Convert sources to VerifiedDocuments and save
        for source in sources:
            # Skip already verified sources
            if source.verification_status == VerificationStatus.VERIFIED:
                continue
                
            # Convert to VerifiedDocument
            doc = VerifiedDocument(
                id=source.id,
                content=source.content,
                title=source.title,
                url=source.url,
                domain=source.domain,
                verification_status=VerificationStatus.VERIFIED,
                metadata=source.metadata
            )
            
            # Save to document store
            document_store.add_document(doc)
            
            # Update source verification status
            source.verification_status = VerificationStatus.VERIFIED
        
        return {"sources": sources}
    
    except Exception as e:
        logger.error(f"Error saving verified sources: {e}")
        return {"sources": sources}


def should_use_sources(state: SectionState) -> Literal["write_section", "search_web_after_verification_failure"]:
    """Decision function that determines if verified sources should be used.
    
    Args:
        state: Current section state.
        
    Returns:
        Next node to visit.
    """
    # Get verification feedback
    verification_feedback = state.get("verification_feedback")
    
    # If no verification feedback or verification passed, use sources
    if not verification_feedback or verification_feedback.verification_passed:
        return "write_section"
    
    # Otherwise, handle verification failure
    return "search_web_after_verification_failure"


async def search_web_after_verification_failure(state: SectionState, config: RunnableConfig):
    """Handle verification failure by generating new search queries.
    
    Args:
        state: Current section state.
        config: Configuration.
        
    Returns:
        Dict with new search queries.
    """
    # Get state
    topic = state["topic"]
    section = state["section"]
    verification_feedback = state["verification_feedback"]
    search_iterations = state["search_iterations"]
    
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries
    
    # Format verification issues for the LLM
    issues_str = "\n".join([
        f"- {issue}" for issue in verification_feedback.verification_issues
    ])
    
    # Prompt for generating better search queries
    system_prompt = f"""You are a research specialist tasked with creating better search queries after verification issues.

TOPIC: {topic}
SECTION: {section.name} - {section.description}

The previous search attempt failed verification due to the following issues:
{issues_str}

Your task is to generate {number_of_queries} new, improved search queries that will likely yield:
1. More reliable sources
2. More accurate information
3. Information that addresses the verification issues

Focus on:
- More specific queries that target authoritative sources
- Queries that include terms like "official", "authoritative", "academic", "peer-reviewed"
- For legal/tax content, target official government sources, legislation, or recognized authorities
"""

    # Generate new queries
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    
    structured_llm = writer_model.with_structured_output(type=List[SearchQuery])
    
    queries_response = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Generate improved search queries that will yield sources that can pass verification.")
    ])
    
    # Return new search queries and increment iterations
    return {
        "search_queries": queries_response,
        "search_iterations": search_iterations + 1
    }