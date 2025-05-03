from typing import Annotated, List, Dict, Any, Optional, TypedDict, Literal, Union
from pydantic import BaseModel, Field
import operator
import datetime
import uuid

from othertales.deepresearch.storage import VerificationStatus

class Source(BaseModel):
    """Represents a verified source of information."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique source identifier")
    title: str = Field(description="Title of the source")
    url: Optional[str] = Field(None, description="URL of the source if available")
    content: str = Field(description="Content from the source")
    domain: Literal["legal", "tax", "general"] = Field("general", description="Domain of the source")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Verification status of the source"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    retrieved_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="When the source was retrieved"
    )

class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources used in this section"
    )
    verified: bool = Field(
        default=False,
        description="Whether this section's content has been verified"
    )

class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class KnowledgeResult(BaseModel):
    """Results from knowledge base lookup."""
    has_relevant_information: bool = Field(
        description="Whether relevant information was found in the knowledge base"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources found in knowledge base"
    )
    relevance_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Relevance score of the information found"
    )
    content_summary: str = Field(
        description="Summary of relevant content from knowledge base"
    )

class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class VerificationFeedback(BaseModel):
    """Feedback from verification process."""
    verification_status: VerificationStatus = Field(
        description="Verification status"
    )
    verification_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details of verification results"
    )
    verification_issues: List[str] = Field(
        default_factory=list,
        description="List of verification issues found"
    )
    verification_passed: bool = Field(
        description="Whether verification passed all checks"
    )
    follow_up_actions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions if verification failed"
    )

class AgentMode(str, Literal["legal", "tax", "general"]):
    """Operation mode for the agent."""
    pass

class ReportStateInput(TypedDict):
    topic: str # Report topic
    mode: Optional[AgentMode] # Agent operation mode
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report
    sources: Optional[List[Source]] # Sources used in the report
    verification_status: Optional[VerificationStatus] # Overall verification status

class ReportState(TypedDict):
    topic: str # Report topic    
    mode: AgentMode # Agent operation mode (legal, tax, general)
    feedback_on_report_plan: str # Feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report
    sources: List[Source] # All sources used in the report
    verification_status: VerificationStatus # Overall verification status
    verification_details: Dict[str, Any] # Details about verification

class SectionState(TypedDict):
    topic: str # Report topic
    mode: AgentMode # Agent operation mode (legal, tax, general)
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    sources: List[Source] # Sources retrieved for this section
    knowledge_results: Optional[KnowledgeResult] # Results from knowledge base lookup
    verification_feedback: Optional[VerificationFeedback] # Feedback from verification process
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    sources: List[Source] # Sources used in the section
