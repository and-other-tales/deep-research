"""
Enhanced research graph with knowledge storage and verification.

This module enhances the standard research graph with knowledge storage and verification
capabilities. It supports domain-specific modes for UK legal and HMRC tax research,
and ensures all information is verified by multiple agents before being included in
the final report.

Main enhancements:
1. Knowledge storage and retrieval before web search
2. Multi-agent verification of all sources
3. Domain-specific legal and tax modes
4. Storage of verified information for future use
"""

from typing import Literal, Optional, Dict, Any, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from othertales.deepresearch.state import (
    ReportStateInput,
    ReportStateOutput,
    Sections,
    ReportState,
    SectionState,
    SectionOutputState,
    Queries,
    Feedback,
    AgentMode,
    Source,
    KnowledgeResult,
    VerificationFeedback
)

from othertales.deepresearch.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions, 
    section_writer_instructions,
    final_section_writer_instructions,
    section_grader_instructions,
    section_writer_inputs
)

from othertales.deepresearch.configuration import Configuration
from othertales.deepresearch.utils import (
    format_sections, 
    get_config_value, 
    get_search_params, 
    select_and_execute_search
)

# Import new components
from othertales.deepresearch.storage import (
    DocumentStore,
    VerificationStatus,
    create_document_store
)
from othertales.deepresearch.knowledge import (
    check_existing_knowledge,
    should_perform_web_search,
    write_section_from_knowledge
)
from othertales.deepresearch.verification_workflow import (
    process_sources,
    verify_sources,
    save_verified_sources,
    should_use_sources,
    search_web_after_verification_failure
)

## Nodes -- 

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the initial report plan with sections.
    
    This node:
    1. Gets configuration for the report structure and search parameters
    2. Generates search queries to gather context for planning
    3. Performs web searches using those queries
    4. Uses an LLM to generate a structured plan with sections
    
    Args:
        state: Current graph state containing the report topic
        config: Configuration for models, search APIs, etc.
        
    Returns:
        Dict containing the generated sections
    """

    # Inputs
    topic = state["topic"]
    mode = state.get("mode", "general")
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Add mode-specific instructions
    mode_instructions = ""
    if mode == "legal":
        mode_instructions = """
This report must focus on UK legal information that is accurate, properly sourced from authoritative legal materials, 
and helpful for legal professionals or individuals seeking legal guidance. 

Your report must:
1. Cite specific legislation (Acts of Parliament, Statutory Instruments) where relevant
2. Reference case law properly when discussing legal precedents
3. Clearly distinguish between different UK jurisdictions (England & Wales, Scotland, Northern Ireland)
4. Include appropriate legal disclaimers
5. Maintain legal accuracy while being accessible to the intended audience
"""
    elif mode == "tax":
        mode_instructions = """
This report must focus on UK tax information that is accurate, compliant with HMRC regulations, and helpful for 
tax professionals or individuals seeking tax guidance.

Your report must:
1. Reference specific tax legislation and HMRC guidance where relevant
2. Include appropriate tax years for all rates, thresholds, and allowances
3. Distinguish between different types of taxation (Income Tax, Corporation Tax, VAT, etc.)
4. Include appropriate tax disclaimers
5. Balance tax compliance requirements with optimization opportunities
"""

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        number_of_queries=number_of_queries
    )
    
    # Add mode-specific instructions if applicable
    if mode_instructions:
        system_instructions_query += f"\n\nADDITIONAL MODE INSTRUCTIONS:\n{mode_instructions}"

    # Generate queries  
    results = await structured_llm.ainvoke([SystemMessage(content=system_instructions_query),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        context=source_str, 
        feedback=feedback
    )
    
    # Add mode-specific instructions if applicable
    if mode_instructions:
        system_instructions_sections += f"\n\nADDITIONAL MODE INSTRUCTIONS:\n{mode_instructions}"

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider, 
                                      max_tokens=20_000, 
                                      thinking={"type": "enabled", "budget_tokens": 16_000})

    else:
        # With other models, thinking tokens are not specifically allocated
        planner_llm = init_chat_model(model=planner_model, 
                                      model_provider=planner_provider,
                                      model_kwargs=planner_model_kwargs)
    
    # Generate the report sections
    structured_llm = planner_llm.with_structured_output(Sections)
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    # Get sections
    sections = report_sections.sections

    # Initialize empty sources list and verification status
    return {
        "sections": sections,
        "sources": [],
        "verification_status": VerificationStatus.PENDING
    }

def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """Get human feedback on the report plan and route to next steps.
    
    This node:
    1. Formats the current report plan for human review
    2. Gets feedback via an interrupt
    3. Routes to either:
       - Section writing if plan is approved
       - Plan regeneration if feedback is provided
    
    Args:
        state: Current graph state with sections to review
        config: Configuration for the workflow
        
    Returns:
        Command to either regenerate plan or start section writing
    """

    # Get sections
    topic = state["topic"]
    mode = state.get("mode", "general")
    sections = state['sections']
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    mode_str = f"Mode: {mode.upper()}\n" if mode != "general" else ""
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\n{mode_str}Topic: {topic}\n\n{sections_str}\n
                        \nDoes the report plan meet your needs?\nPass 'true' to approve the report plan.\nOr, provide feedback to regenerate the report plan:"""
    
    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if isinstance(feedback, bool) and feedback is True:
        # Treat this as approve and kick off section writing
        return Command(goto=[
            Send("build_section_with_web_research", {
                "topic": topic, 
                "mode": mode,
                "section": s, 
                "search_iterations": 0
            }) 
            for s in sections 
            if s.research
        ])
    
    # If the user provides feedback, regenerate the report plan 
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": feedback})
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")
    
async def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for researching a specific section.
    
    This node uses an LLM to generate targeted search queries based on the 
    section topic and description.
    
    Args:
        state: Current state containing section details
        config: Configuration including number of queries to generate
        
    Returns:
        Dict containing the generated search queries
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    mode = state.get("mode", "general")

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries 
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)

    # Add mode-specific instructions
    mode_instructions = ""
    if mode == "legal":
        mode_instructions = """
Focus your search queries on:
1. Official UK government and legal websites (legislation.gov.uk, gov.uk, judiciary.uk)
2. Reputable legal publishers and databases (Westlaw, LexisNexis, Practical Law)
3. UK court decisions and case law
4. Law society and bar council publications
5. Academic legal journals and university law department publications

Include specific legal terminology and references to relevant Acts, regulations, or legal principles.
"""
    elif mode == "tax":
        mode_instructions = """
Focus your search queries on:
1. Official HMRC and UK government websites (gov.uk, HMRC manuals)
2. Reputable tax publishers and professional bodies (ICAEW, CIOT, ICAS)
3. Tax legislation and Finance Acts
4. HMRC guidance, tax bulletins and briefs
5. Recognized tax journals and publications

Include specific tax terminology, relevant tax years, and references to specific tax provisions or HMRC guidance.
"""

    # Format system instructions
    system_instructions = query_writer_instructions.format(
        topic=topic, 
        section_topic=section.description, 
        number_of_queries=number_of_queries
    )
    
    # Add mode-specific instructions if applicable
    if mode_instructions:
        system_instructions += f"\n\nADDITIONAL MODE INSTRUCTIONS:\n{mode_instructions}"

    # Generate queries  
    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                     HumanMessage(content="Generate search queries on the provided topic.")])

    return {"search_queries": queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """Execute web searches for the section queries.
    
    This node:
    1. Takes the generated queries
    2. Executes searches using configured search API
    3. Formats results into usable context
    
    Args:
        state: Current state with search queries
        config: Search API configuration
        
    Returns:
        Dict with search results and updated iteration count
    """

    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = configurable.search_api_config or {}  # Get the config dict, default to empty
    params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    source_str = await select_and_execute_search(search_api, query_list, params_to_pass)

    return {"source_str": source_str, "search_iterations": state["search_iterations"] + 1}

async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report and evaluate if more research is needed.
    
    This node:
    1. Writes section content using search results
    2. Evaluates the quality of the section
    3. Either:
       - Completes the section if quality passes
       - Triggers more research if quality fails
    
    Args:
        state: Current state with search results and section info
        config: Configuration for writing and evaluation
        
    Returns:
        Command to either complete section or do more research
    """

    # Get state 
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]
    sources = state.get("sources", [])
    mode = state.get("mode", "general")

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Add mode-specific instructions
    mode_instructions = ""
    if mode == "legal":
        mode_instructions = """
Additional Legal Writing Guidelines:
- Maintain legal accuracy and precision
- Cite specific legislation (Acts, Statutory Instruments) with proper references
- Reference case law using correct citation format
- Clearly distinguish between different UK jurisdictions when relevant
- Include appropriate legal disclaimers if providing guidance
- Use proper legal terminology while remaining accessible
"""
    elif mode == "tax":
        mode_instructions = """
Additional Tax Writing Guidelines:
- Ensure all tax information is accurate and compliant with current HMRC guidance
- Specify relevant tax years for all rates, thresholds, and allowances
- Reference specific HMRC manuals, guidance, or legislation where appropriate
- Clearly distinguish between different tax regimes when relevant
- Include appropriate tax disclaimers if providing guidance
- Balance technical accuracy with accessibility
"""

    # Format system instructions
    section_writer_instructions_enhanced = section_writer_instructions
    if mode_instructions:
        section_writer_instructions_enhanced += f"\n\n{mode_instructions}"
        
    section_writer_inputs_formatted = section_writer_inputs.format(
        topic=topic, 
        section_name=section.name, 
        section_topic=section.description, 
        context=source_str, 
        section_content=section.content
    )

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 

    section_content = await writer_model.ainvoke([
        SystemMessage(content=section_writer_instructions_enhanced),
        HumanMessage(content=section_writer_inputs_formatted)
    ])
    
    # Write content to the section object  
    section.content = section_content.content

    # Grade prompt 
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                          "If the grade is 'pass', return empty strings for all follow-up queries. "
                          "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic, 
        section_topic=section.description,
        section=section.content, 
        number_of_follow_up_queries=configurable.number_of_queries
    )

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    planner_model_kwargs = get_config_value(configurable.planner_model_kwargs or {})

    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider, 
            max_tokens=20_000, 
            thinking={"type": "enabled", "budget_tokens": 16_000}
        ).with_structured_output(Feedback)
    else:
        reflection_model = init_chat_model(
            model=planner_model, 
            model_provider=planner_provider, 
            model_kwargs=planner_model_kwargs
        ).with_structured_output(Feedback)
    
    # Generate feedback
    feedback = await reflection_model.ainvoke([
        SystemMessage(content=section_grader_instructions_formatted),
        HumanMessage(content=section_grader_message)
    ])

    # Update section with sources
    section.sources = sources
    
    # If the section is passing or the max search depth is reached, publish the section to completed sections 
    if feedback.grade == "pass" or state["search_iterations"] >= configurable.max_search_depth:
        # If verification is required and sources exist, mark section for verification
        if configurable.verification_required and sources:
            section.verified = False
            return Command(
                update={"section": section},
                goto="verify_sources"
            )
        else:
            # Skip verification
            section.verified = True
            return Command(
                update={"completed_sections": [section], "sources": sources},
                goto=END
            )

    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web"
        )
    
async def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write sections that don't require research using completed sections as context.
    
    This node handles sections like conclusions or summaries that build on
    the researched sections rather than requiring direct research.
    
    Args:
        state: Current state with completed sections as context
        config: Configuration for the writing model
        
    Returns:
        Dict containing the newly written section
    """

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state 
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    mode = state.get("mode", "general")
    sources = state.get("sources", [])
    
    # Add mode-specific instructions
    mode_instructions = ""
    if mode == "legal":
        mode_instructions = """
Additional Legal Writing Guidelines for Summary/Conclusion:
- Maintain legal accuracy while summarizing key points
- Include appropriate disclaimers about legal advice
- Summarize the key legal principles, statutes, or cases referenced
- Highlight jurisdictional considerations if relevant
- Present balanced legal perspectives when appropriate
"""
    elif mode == "tax":
        mode_instructions = """
Additional Tax Writing Guidelines for Summary/Conclusion:
- Maintain tax accuracy while summarizing key points
- Include appropriate disclaimers about tax advice
- Summarize key tax rules, rates, or HMRC guidance referenced
- Highlight tax year considerations and future changes if relevant
- Balance compliance requirements with optimization opportunities
"""

    # Format system instructions
    final_instructions = final_section_writer_instructions
    if mode_instructions:
        final_instructions += f"\n\n{mode_instructions}"
        
    system_instructions = final_instructions.format(
        topic=topic, 
        section_name=section.name, 
        section_topic=section.description, 
        context=completed_report_sections
    )

    # Generate section  
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model_kwargs = get_config_value(configurable.writer_model_kwargs or {})
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    section_content = await writer_model.ainvoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a report section based on the provided sources.")
    ])
    
    # Write content to section 
    section.content = section_content.content
    
    # Mark as verified (summary sections don't need verification)
    section.verified = True
    section.sources = sources

    # Write the updated section to completed sections
    return {"completed_sections": [section], "sources": sources}

def gather_completed_sections(state: ReportState):
    """Format completed sections as context for writing final sections.
    
    This node takes all completed research sections and formats them into
    a single context string for writing summary sections.
    
    Args:
        state: Current state with completed sections
        
    Returns:
        Dict with formatted sections as context
    """

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}

def compile_final_report(state: ReportState):
    """Compile all sections into the final report.
    
    This node:
    1. Gets all completed sections
    2. Orders them according to original plan
    3. Combines them into the final report
    
    Args:
        state: Current state with all completed sections
        
    Returns:
        Dict containing the complete report
    """

    # Get sections
    sections = state["sections"]
    completed_sections = {s.name: s for s in state["completed_sections"]}
    sources = state.get("sources", [])

    # Update sections with completed content while maintaining original order
    for section in sections:
        if section.name in completed_sections:
            completed_section = completed_sections[section.name]
            section.content = completed_section.content
            section.sources = completed_section.sources
            section.verified = completed_section.verified

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    # Collect all unique sources
    all_sources = []
    seen_source_ids = set()
    
    for section in sections:
        for source in section.sources:
            if source.id not in seen_source_ids:
                seen_source_ids.add(source.id)
                all_sources.append(source)

    # Determine overall verification status
    all_verified = all(section.verified for section in sections)
    verification_status = VerificationStatus.VERIFIED if all_verified else VerificationStatus.PARTIAL

    return {
        "final_report": all_sections,
        "sources": all_sources,
        "verification_status": verification_status
    }

def initiate_final_section_writing(state: ReportState):
    """Create parallel tasks for writing non-research sections.
    
    This edge function identifies sections that don't need research and
    creates parallel writing tasks for each one.
    
    Args:
        state: Current state with all sections and research context
        
    Returns:
        List of Send commands for parallel section writing
    """

    # Get mode
    mode = state.get("mode", "general")
    sources = state.get("sources", [])
    
    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send(
            "write_final_sections", 
            {
                "topic": state["topic"], 
                "mode": mode,
                "section": s, 
                "report_sections_from_research": state["report_sections_from_research"],
                "sources": sources
            }
        ) 
        for s in state["sections"] 
        if not s.research
    ]

# Create the enhanced research graph
def create_enhanced_graph():
    """Create the enhanced research graph with knowledge storage and verification."""
    
    # Report section sub-graph -- 
    
    # Add nodes 
    section_builder = StateGraph(SectionState, output=SectionOutputState)
    
    # Knowledge retrieval nodes
    section_builder.add_node("check_existing_knowledge", check_existing_knowledge)
    section_builder.add_node("write_section_from_knowledge", write_section_from_knowledge)
    
    # Standard research nodes
    section_builder.add_node("generate_queries", generate_queries)
    section_builder.add_node("search_web", search_web)
    section_builder.add_node("write_section", write_section)
    
    # Verification nodes
    section_builder.add_node("process_sources", process_sources)
    section_builder.add_node("verify_sources", verify_sources)
    section_builder.add_node("save_verified_sources", save_verified_sources)
    section_builder.add_node("search_web_after_verification_failure", search_web_after_verification_failure)
    
    # Add edges
    section_builder.add_edge(START, "check_existing_knowledge")
    
    # Knowledge retrieval flow
    section_builder.add_conditional_edges(
        "check_existing_knowledge",
        should_perform_web_search,
        {
            "search_web": "generate_queries",
            "write_section_from_knowledge": "write_section_from_knowledge"
        }
    )
    section_builder.add_edge("write_section_from_knowledge", END)
    
    # Standard research flow
    section_builder.add_edge("generate_queries", "search_web")
    section_builder.add_edge("search_web", "process_sources")
    section_builder.add_edge("process_sources", "write_section")
    
    # Verification flow
    section_builder.add_edge("write_section", "verify_sources")
    section_builder.add_edge("verify_sources", "save_verified_sources")
    
    section_builder.add_conditional_edges(
        "save_verified_sources",
        should_use_sources,
        {
            "write_section": END,
            "search_web_after_verification_failure": "search_web_after_verification_failure"
        }
    )
    
    section_builder.add_edge("search_web_after_verification_failure", "search_web")
    
    # Outer graph for initial report plan compiling results from each section -- 
    
    # Add nodes
    builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
    builder.add_node("generate_report_plan", generate_report_plan)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("build_section_with_web_research", section_builder.compile())
    builder.add_node("gather_completed_sections", gather_completed_sections)
    builder.add_node("write_final_sections", write_final_sections)
    builder.add_node("compile_final_report", compile_final_report)
    
    # Add edges
    builder.add_edge(START, "generate_report_plan")
    builder.add_edge("generate_report_plan", "human_feedback")
    builder.add_edge("build_section_with_web_research", "gather_completed_sections")
    builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)
    
    return builder.compile()

# Initialize the graph
graph = create_enhanced_graph()