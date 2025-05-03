"""
Verification agent system with 3-agent verification protocol.

This module implements a rigorous verification system for document content
using three independent verification agents. Each agent evaluates the content
for accuracy, relevance, and trustworthiness. A document is only considered
verified when all three agents approve it.

Main components:
- VerificationAgent: Base class for verification agents
- FactCheckAgent: Verifies factual accuracy of content
- SourceEvaluationAgent: Evaluates trustworthiness of sources
- ConsistencyAgent: Checks for internal consistency and logical coherence
- VerificationCoordinator: Manages the verification process across agents
"""

import os
import uuid
import asyncio
import datetime
from typing import Any, Dict, List, Optional, Union, Literal, Tuple

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

# Import storage classes
from othertales.deepresearch.storage import (
    VerificationStatus, VerifiedDocument, DocumentStore
)

# Configuration
from othertales.deepresearch.configuration import Configuration
from othertales.deepresearch.utils import get_config_value

# Configure logging
import logging
logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Output of a verification agent."""
    agent_id: str = Field(description="Unique identifier for the agent")
    document_id: str = Field(description="ID of the document being verified")
    status: Literal["pass", "fail"] = Field(description="Overall verification result")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Detailed reasoning for the verification result")
    issues: List[str] = Field(description="List of specific issues identified (if any)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class VerificationType(BaseModel):
    """Type of verification being performed."""
    factual_accuracy: bool = True
    source_reliability: bool = True
    internal_consistency: bool = True
    legal_correctness: bool = False
    tax_compliance: bool = False


# Base Verification Agent
class VerificationAgent:
    """Base class for verification agents."""
    
    def __init__(
        self,
        agent_id: str = None,
        agent_name: str = "Generic Verification Agent",
        model_provider: str = "anthropic",
        model_name: str = "claude-3-5-sonnet-latest",
        model_kwargs: Dict[str, Any] = None,
    ):
        """Initialize a verification agent.
        
        Args:
            agent_id: Unique identifier for the agent.
            agent_name: Human-readable name for the agent.
            model_provider: Provider of the LLM.
            model_name: Name of the model to use.
            model_kwargs: Additional arguments for the model.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_name = agent_name
        self.model_provider = model_provider
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        
        # Initialize LLM
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            model_kwargs=self.model_kwargs
        )
        
        # Setup output parser
        self.parser = PydanticOutputParser(pydantic_object=VerificationResult)
        
    async def verify(
        self, 
        document: VerifiedDocument,
        verification_type: VerificationType = None,
        reference_documents: List[VerifiedDocument] = None,
    ) -> VerificationResult:
        """Verify a document.
        
        Args:
            document: Document to verify.
            verification_type: Type of verification to perform.
            reference_documents: Optional reference documents for verification.
            
        Returns:
            Verification result.
        """
        # Default verification type if not provided
        if verification_type is None:
            verification_type = VerificationType()
            
        # Get prompt for verification
        prompt = self._get_verification_prompt(
            document, verification_type, reference_documents
        )
        
        # Run verification
        result = await self._run_verification(prompt, document)
        
        return result
    
    def _get_verification_prompt(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType,
        reference_documents: Optional[List[VerifiedDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Get the prompt for verification.
        
        Args:
            document: Document to verify.
            verification_type: Type of verification to perform.
            reference_documents: Optional reference documents for verification.
            
        Returns:
            List of messages for the prompt.
        """
        # Base system prompt
        system_prompt = f"""You are {self.agent_name}, a critical verification agent tasked with rigorously evaluating document content for accuracy and reliability.

Your task is to verify the following document and provide a clear PASS or FAIL determination.

Verification Types Requested:
- Factual Accuracy: {verification_type.factual_accuracy}
- Source Reliability: {verification_type.source_reliability}
- Internal Consistency: {verification_type.internal_consistency}
- Legal Correctness: {verification_type.legal_correctness}
- Tax Compliance: {verification_type.tax_compliance}

DO NOT MAKE ASSUMPTIONS about the correctness of information. Your default stance should be skepticism.
You must clearly justify your reasoning with specific examples from the text.

Your output must be a valid JSON object matching this schema:
{self.parser.get_format_instructions()}
"""

        # Format document content
        document_content = f"""
DOCUMENT TITLE: {document.title}
DOCUMENT ID: {document.id}
SOURCE URL: {document.url or 'Not specified'}
DOMAIN: {document.domain}

CONTENT:
{document.content}
"""

        # Format reference documents if provided
        reference_content = ""
        if reference_documents:
            reference_content = "REFERENCE DOCUMENTS FOR VERIFICATION:\n\n"
            for i, ref_doc in enumerate(reference_documents, 1):
                reference_content += f"""
REFERENCE {i}:
TITLE: {ref_doc.title}
URL: {ref_doc.url or 'Not specified'}
CONTENT:
{ref_doc.content[:1000]}... [truncated]
"""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": document_content},
        ]
        
        # Add reference content if available
        if reference_content:
            messages.append({"role": "user", "content": reference_content})
            
        # Add verification instructions
        verification_instructions = """
Please verify this document carefully and provide your determination.
You must include a clear "pass" or "fail" status in your response.
Your reasoning should be detailed and specific to the content.
If you identify issues, list them specifically.
"""
        messages.append({"role": "user", "content": verification_instructions})
        
        return messages
    
    async def _run_verification(
        self,
        prompt: List[Dict[str, Any]],
        document: VerifiedDocument,
    ) -> VerificationResult:
        """Run the verification with the LLM.
        
        Args:
            prompt: Prompt messages.
            document: Document being verified.
            
        Returns:
            Verification result.
        """
        try:
            # Convert prompt to message objects
            messages = []
            for msg in prompt:
                if msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            # Run the model
            response = await self.llm.ainvoke(messages)
            
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
                
                # Ensure the document ID is set correctly
                result.document_id = document.id
                result.agent_id = self.agent_id
                
                return result
            
            except Exception as e:
                # If parsing fails, create a minimal result
                logger.error(f"Failed to parse verification result: {e}")
                return VerificationResult(
                    agent_id=self.agent_id,
                    document_id=document.id,
                    status="fail",
                    confidence=0.0,
                    reasoning=f"Failed to parse verification result: {str(e)}. Raw response: {response.content}",
                    issues=["Verification result parsing failed"]
                )
        except Exception as e:
            # Handle any errors in the verification process
            logger.error(f"Verification failed: {e}")
            return VerificationResult(
                agent_id=self.agent_id,
                document_id=document.id,
                status="fail",
                confidence=0.0,
                reasoning=f"Verification process failed: {str(e)}",
                issues=["Verification process failed"]
            )


class FactCheckAgent(VerificationAgent):
    """Agent specializing in factual accuracy verification."""
    
    def __init__(
        self,
        agent_id: str = None,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-5-sonnet-latest",
        model_kwargs: Dict[str, Any] = None,
    ):
        """Initialize a fact-checking agent."""
        super().__init__(
            agent_id=agent_id,
            agent_name="Factual Accuracy Verification Agent",
            model_provider=model_provider,
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
    def _get_verification_prompt(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType,
        reference_documents: Optional[List[VerifiedDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Get specialized fact-checking prompt."""
        # Get base prompt
        messages = super()._get_verification_prompt(
            document, verification_type, reference_documents
        )
        
        # Add fact-checking specific instructions
        fact_check_instructions = """
SPECIFIC FACT-CHECKING INSTRUCTIONS:

1. Identify all factual claims in the document
2. Check these claims against reference materials or your knowledge base
3. Note any factual errors, unsupported claims, or misleading statements
4. Be especially attentive to:
   - Dates, numbers, statistics, and quantitative claims
   - Names, titles, and affiliations
   - Historical events and timelines
   - Scientific or technical statements
   - Legal and regulatory references
5. Distinguish between verified facts, likely facts, and unsupported assertions
6. For legal and tax content, verify that statute references, case citations, and regulatory guidance are accurate

Your primary focus is on FACTUAL ACCURACY. A document passes only if its factual claims are verifiably correct or properly qualified when uncertain.
"""
        
        # Insert fact-checking instructions before the verification instructions
        messages.insert(-1, {"role": "user", "content": fact_check_instructions})
        
        return messages


class SourceEvaluationAgent(VerificationAgent):
    """Agent specializing in evaluating source reliability and credibility."""
    
    def __init__(
        self,
        agent_id: str = None,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-5-sonnet-latest",
        model_kwargs: Dict[str, Any] = None,
    ):
        """Initialize a source evaluation agent."""
        super().__init__(
            agent_id=agent_id,
            agent_name="Source Reliability Verification Agent",
            model_provider=model_provider,
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
    def _get_verification_prompt(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType,
        reference_documents: Optional[List[VerifiedDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Get specialized source evaluation prompt."""
        # Get base prompt
        messages = super()._get_verification_prompt(
            document, verification_type, reference_documents
        )
        
        # Add source evaluation specific instructions
        source_eval_instructions = """
SPECIFIC SOURCE EVALUATION INSTRUCTIONS:

1. Evaluate the credibility and reliability of the document's source
2. Check for:
   - Official sources (government websites, academic institutions, recognized authorities)
   - Reputable publications with editorial standards
   - Primary vs. secondary sources
   - Potential bias or conflicts of interest
   - Currency and timeliness of information
3. For legal and tax content, prioritize:
   - Primary legal sources (legislation, case law, official government guidance)
   - Recognized legal publishers and databases
   - Up-to-date information (verify that laws haven't been superseded)
   - Official HMRC/government documentation
4. Consider domain-specific source reliability factors:
   - For legal content: court websites, legislation.gov.uk, reputable law firms, legal journals
   - For tax content: HMRC official guidance, recognized tax advisory bodies, accounting institutes

For content from unofficial sources, verify if the information matches authoritative sources.

Your primary focus is on SOURCE RELIABILITY. A document passes only if its content comes from or is verifiable against trustworthy, authoritative sources.
"""
        
        # Insert source evaluation instructions before the verification instructions
        messages.insert(-1, {"role": "user", "content": source_eval_instructions})
        
        return messages


class ConsistencyAgent(VerificationAgent):
    """Agent specializing in checking internal consistency and logical coherence."""
    
    def __init__(
        self,
        agent_id: str = None,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-5-sonnet-latest",
        model_kwargs: Dict[str, Any] = None,
    ):
        """Initialize a consistency checking agent."""
        super().__init__(
            agent_id=agent_id,
            agent_name="Consistency and Coherence Verification Agent",
            model_provider=model_provider,
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
    def _get_verification_prompt(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType,
        reference_documents: Optional[List[VerifiedDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Get specialized consistency checking prompt."""
        # Get base prompt
        messages = super()._get_verification_prompt(
            document, verification_type, reference_documents
        )
        
        # Add consistency checking specific instructions
        consistency_instructions = """
SPECIFIC CONSISTENCY CHECKING INSTRUCTIONS:

1. Evaluate the document for internal consistency and logical coherence
2. Check for:
   - Contradictory statements or positions
   - Logical fallacies and reasoning errors
   - Consistency of terminology usage
   - Appropriate progression of arguments or explanations
   - Missing steps in logical sequences
3. For legal and tax content, additionally verify:
   - Consistent application of legal principles
   - Proper logical flow from legal premises to conclusions
   - Correct application of precedent and statutory interpretation
   - Consistency with generally accepted legal or tax principles
   - Appropriate qualifications for jurisdictional differences
4. Check if conclusions properly follow from premises and evidence presented
5. Verify that different sections of the document maintain consistency in their assertions

Your primary focus is on LOGICAL CONSISTENCY. A document passes only if it maintains internal consistency, avoids contradictions, and follows sound logical reasoning.
"""
        
        # Insert consistency instructions before the verification instructions
        messages.insert(-1, {"role": "user", "content": consistency_instructions})
        
        return messages


class LegalCorrectnessAgent(VerificationAgent):
    """Agent specializing in verifying legal correctness for UK law."""
    
    def __init__(
        self,
        agent_id: str = None,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-sonnet-latest",
        model_kwargs: Dict[str, Any] = None,
    ):
        """Initialize a legal correctness agent."""
        super().__init__(
            agent_id=agent_id,
            agent_name="UK Legal Correctness Verification Agent",
            model_provider=model_provider,
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
    def _get_verification_prompt(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType,
        reference_documents: Optional[List[VerifiedDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Get specialized legal correctness prompt."""
        # Get base prompt
        messages = super()._get_verification_prompt(
            document, verification_type, reference_documents
        )
        
        # Add legal correctness specific instructions
        legal_instructions = """
SPECIFIC UK LEGAL CORRECTNESS INSTRUCTIONS:

1. Verify accuracy of UK legal information, including:
   - Statute citations (Acts of Parliament, Statutory Instruments)
   - Case law references (ensure correct case names, citations, and interpretations)
   - Legal principles and their applications
   - Jurisdictional boundaries (England & Wales, Scotland, Northern Ireland)
   - EU law applicability post-Brexit
   - Procedural rules and timelines

2. Check currency of legal information:
   - Verify legislation hasn't been repealed or amended
   - Ensure case law hasn't been overturned or distinguished
   - Confirm that legal procedures are current

3. Evaluate proper legal reasoning:
   - Correct application of precedent
   - Appropriate statutory interpretation
   - Valid legal analysis and conclusions
   - Proper consideration of relevant legal authorities

4. Verify accuracy of specialized legal domains if present:
   - Criminal law (elements of offenses, sentencing guidelines)
   - Civil procedure (court processes, time limits)
   - Contract law (formation, terms, remedies)
   - Property law (land law, tenancies, conveyancing)
   - Family law (divorce, children, financial remedies)
   - Employment law (rights, obligations, tribunals)

5. Ensure proper representation of:
   - Burden of proof standards
   - Legal presumptions
   - Court hierarchies and jurisdiction
   - Legal terminology usage

Your primary focus is on LEGAL CORRECTNESS in the UK context. A document passes only if its legal content accurately represents current UK law, correctly applies legal principles, and provides accurate guidance within the UK legal system.
"""
        
        # Insert legal instructions before the verification instructions
        messages.insert(-1, {"role": "user", "content": legal_instructions})
        
        return messages


class TaxComplianceAgent(VerificationAgent):
    """Agent specializing in verifying tax compliance with HMRC regulations."""
    
    def __init__(
        self,
        agent_id: str = None,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-sonnet-latest",
        model_kwargs: Dict[str, Any] = None,
    ):
        """Initialize a tax compliance agent."""
        super().__init__(
            agent_id=agent_id,
            agent_name="HMRC Tax Compliance Verification Agent",
            model_provider=model_provider,
            model_name=model_name,
            model_kwargs=model_kwargs
        )
        
    def _get_verification_prompt(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType,
        reference_documents: Optional[List[VerifiedDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Get specialized tax compliance prompt."""
        # Get base prompt
        messages = super()._get_verification_prompt(
            document, verification_type, reference_documents
        )
        
        # Add tax compliance specific instructions
        tax_instructions = """
SPECIFIC HMRC TAX COMPLIANCE INSTRUCTIONS:

1. Verify accuracy of UK tax information, including:
   - Tax legislation and HMRC guidelines
   - Tax rates, thresholds, and allowances
   - Filing requirements and deadlines
   - Tax calculation methodologies
   - Eligibility criteria for reliefs and exemptions
   - Compliance obligations and penalties

2. Check currency of tax information:
   - Ensure tax rates and thresholds are current for the correct tax year
   - Verify that guidance hasn't been superseded by new legislation or HMRC policies
   - Confirm that tax procedures reflect current HMRC processes

3. Evaluate proper tax guidance:
   - Correct application of tax principles
   - Appropriate distinction between mandatory requirements and planning opportunities
   - Valid tax analysis and conclusions
   - Proper consideration of anti-avoidance rules

4. Verify accuracy of specialized tax domains if present:
   - Income Tax and National Insurance
   - Corporation Tax
   - Capital Gains Tax
   - Inheritance Tax
   - VAT
   - Stamp Duty Land Tax / LBTT / LTT
   - Tax credits and benefits
   - Self-assessment procedures

5. Ensure proper representation of:
   - Tax authority positions and policies
   - Taxpayer obligations and rights
   - Compliance requirements
   - Record-keeping obligations
   - Risk areas and common pitfalls

Your primary focus is on TAX COMPLIANCE in the UK context. A document passes only if its tax content accurately represents current UK tax law and HMRC requirements, correctly applies tax principles, and provides accurate guidance that would not put taxpayers at risk of non-compliance with HMRC regulations.
"""
        
        # Insert tax instructions before the verification instructions
        messages.insert(-1, {"role": "user", "content": tax_instructions})
        
        return messages


class VerificationCoordinator:
    """Coordinates verification across multiple agents."""
    
    def __init__(
        self,
        document_store: DocumentStore,
        config: Optional[RunnableConfig] = None,
    ):
        """Initialize the verification coordinator.
        
        Args:
            document_store: Document store for persisting verification results.
            config: Optional configuration for the verification process.
        """
        self.document_store = document_store
        self.config = config
        
        # Initialize default agents
        self._init_agents(config)
        
    def _init_agents(self, config: Optional[RunnableConfig] = None):
        """Initialize verification agents based on configuration."""
        # Get configuration
        configurable = None
        if config:
            configurable = Configuration.from_runnable_config(config)
        
        # Default model settings if configuration not available
        model_provider = "anthropic"
        fact_check_model = "claude-3-5-sonnet-latest"
        source_eval_model = "claude-3-5-sonnet-latest"
        consistency_model = "claude-3-5-sonnet-latest"
        legal_model = "claude-3-sonnet-latest"
        tax_model = "claude-3-sonnet-latest"
        
        # Use configuration if available
        if configurable:
            model_provider = get_config_value(configurable.verification_provider or model_provider)
            fact_check_model = get_config_value(configurable.fact_check_model or fact_check_model)
            source_eval_model = get_config_value(configurable.source_eval_model or source_eval_model)
            consistency_model = get_config_value(configurable.consistency_model or consistency_model)
            legal_model = get_config_value(configurable.legal_model or legal_model)
            tax_model = get_config_value(configurable.tax_model or tax_model)
        
        # Initialize standard agents
        self.fact_check_agent = FactCheckAgent(
            model_provider=model_provider,
            model_name=fact_check_model
        )
        self.source_eval_agent = SourceEvaluationAgent(
            model_provider=model_provider,
            model_name=source_eval_model
        )
        self.consistency_agent = ConsistencyAgent(
            model_provider=model_provider,
            model_name=consistency_model
        )
        
        # Initialize specialized agents
        self.legal_agent = LegalCorrectnessAgent(
            model_provider=model_provider,
            model_name=legal_model
        )
        self.tax_agent = TaxComplianceAgent(
            model_provider=model_provider,
            model_name=tax_model
        )
        
    async def verify_document(
        self,
        document: VerifiedDocument,
        verification_type: VerificationType = None,
        reference_documents: List[VerifiedDocument] = None,
    ) -> Tuple[bool, Dict[str, VerificationResult]]:
        """Verify a document using all required agents based on verification type.
        
        Args:
            document: Document to verify.
            verification_type: Type of verification to perform.
            reference_documents: Optional reference documents for verification.
            
        Returns:
            Tuple containing:
            - Overall verification result (True if passed all verifications, False otherwise)
            - Dictionary of verification results from each agent
        """
        # Default verification type if not provided
        if verification_type is None:
            verification_type = VerificationType()
            
        # Select agents based on verification type
        agents = []
        if verification_type.factual_accuracy:
            agents.append(self.fact_check_agent)
        if verification_type.source_reliability:
            agents.append(self.source_eval_agent)
        if verification_type.internal_consistency:
            agents.append(self.consistency_agent)
        if verification_type.legal_correctness:
            agents.append(self.legal_agent)
        if verification_type.tax_compliance:
            agents.append(self.tax_agent)
            
        # Run verifications concurrently
        tasks = [
            agent.verify(document, verification_type, reference_documents)
            for agent in agents
        ]
        verification_results = await asyncio.gather(*tasks)
        
        # Collect results
        results_dict = {
            agent.agent_name: result
            for agent, result in zip(agents, verification_results)
        }
        
        # Check if all verifications passed
        all_passed = all(result.status == "pass" for result in verification_results)
        
        # Update document verification status in the store
        if all_passed:
            document_status = VerificationStatus.VERIFIED
        else:
            document_status = VerificationStatus.REJECTED
            
        # List agents that verified the document
        verification_agents = [agent.agent_id for agent in agents]
        
        # Collect verification details
        verification_details = {
            agent.agent_name: {
                "status": result.status,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "issues": result.issues
            }
            for agent, result in zip(agents, verification_results)
        }
            
        # Update document store
        await self._update_document_status(
            document.id,
            document_status,
            verification_agents,
            verification_details
        )
        
        return all_passed, results_dict
    
    async def _update_document_status(
        self,
        document_id: str,
        status: VerificationStatus,
        verification_agents: List[str],
        verification_details: Dict[str, Any]
    ) -> bool:
        """Update the verification status of a document in the store.
        
        Args:
            document_id: ID of the document to update.
            status: New verification status.
            verification_agents: IDs of agents that participated in verification.
            verification_details: Details of the verification process.
            
        Returns:
            Whether the update was successful.
        """
        coordinator_id = f"coordinator-{uuid.uuid4()}"
        
        # Create a summary of the verification
        verification_summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "final_status": status.value,
            "verification_agents": verification_agents,
            "verification_details": verification_details
        }
        
        # Update the document store
        return self.document_store.update_verification_status(
            document_id, status, coordinator_id, verification_summary
        )


# Factory function to create verification coordinator
def create_verification_coordinator(
    document_store: DocumentStore,
    config: Optional[RunnableConfig] = None,
) -> VerificationCoordinator:
    """Create a verification coordinator instance.
    
    Args:
        document_store: Document store for persisting verification results.
        config: Optional configuration for the verification process.
        
    Returns:
        VerificationCoordinator instance.
    """
    try:
        return VerificationCoordinator(document_store, config)
    except Exception as e:
        logger.error(f"Failed to create verification coordinator: {e}")
        raise