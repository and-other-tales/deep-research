# Deep Research w/ Deeper Verification


## 🌟 Key Features

### Enhanced Capabilities
- **Knowledge Storage System**: Vector database and knowledge graph for persistent information
- **Multi-Agent Verification**: 3-agent verification protocol ensures high accuracy
- **Domain-Specific Modes**: Specialized modes for Legal & Business Affairs & Accounting
- **Knowledge-First Research**: Checks existing verified knowledge before external search

### Standard Features
- **Multi-Agent Architecture**: Parallel research with supervisor and researcher agents
- **Workflow Implementation**: Sequential research with reflection and human feedback
- **Multiple Search Tools**: Tavily, Perplexity, Exa, ArXiv, PubMed, and more
- **Customizable Models**: Configure different models for different stages of research
- **Cloud-Ready**: Designed for deployment on Google Cloud Run

![workflow-overview](https://github.com/user-attachments/assets/a171660d-b735-4587-ab2f-cd771f773756)

## 🚀 Quickstart

Clone the repository:
```bash
git clone https://github.com/and-other-tales/deep-research.git
cd deep-research
```

Then edit the `.env` file to customize the environment variables:
```bash
cp .env.example .env
```

Launch the assistant with the LangGraph server locally:

#### Mac
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

#### Windows / Linux
```powershell
# Install dependencies 
pip install -e .
pip install -U "langgraph-cli[inmem]" 

# Start the LangGraph server
langgraph dev
```

Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

## 🔍 Usage Examples

### Knowledge-Enhanced Research

Provide a topic and select a domain mode:

```json
{
  "topic": "Check Corporation Tax Return",
  "mode": "tax"
}
```

The system will:
1. Check existing knowledge in the document store
2. Plan the report based on the topic and tax mode
3. Present a plan for approval
4. After approval, research and write each section
5. Verify all information through the multi-agent verification system
6. Store verified information for future use
7. Present a comprehensive report with citations

### UK Legal Research Mode

```json
{
  "topic": "Provide Information on Derivative and Transformative Works",
  "mode": "legal"
}
```

The legal mode:
- Targets official legal sources (legislation.gov.uk, court websites)
- Verifies jurisdictional applicability
- Checks currency of legal information
- Ensures proper citations of legislation and case law

### General Research Mode

```json
{
  "topic": "Machine learning applications in finance",
  "mode": "general"
}
```

## 📚 Implementation Options

### 1. Enhanced Graph with Knowledge Storage

The enhanced graph implementation (`enhanced_graph.py`) includes:
- Knowledge retrieval before web search
- Multi-agent verification of sources
- Domain-specific research modes
- Storage of verified information

### 2. Standard Graph-based Workflow

The standard graph implementation (`graph.py`) follows a structured plan-and-execute workflow:
- Planning phase with human feedback
- Sequential research with reflection
- Section-specific research
- Support for multiple search tools

### 3. Multi-Agent Implementation

The multi-agent implementation (`multi_agent.py`) uses a supervisor-researcher architecture:
- Supervisor agent manages the overall process
- Researcher agents work in parallel on sections
- Faster report generation with parallel processing

## 🔧 Configuration

### Environment Variables

#### Knowledge Storage
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/deepresearch
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

#### LLM Providers
```
LLM_PROVIDER=anthropic  # Options: openai, anthropic, azure, bedrock, huggingface, huggingface-endpoint, google
```

#### Domain Mode
```
AGENT_MODE=general  # Options: legal, tax, general
```

#### Verification Settings
```
VERIFICATION_REQUIRED=true
MIN_VERIFICATION_CONFIDENCE=0.7
```

#### Server Configuration
```
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
INTERNAL_HOST=127.0.0.1
LANGGRAPH_PORT=8081
```

### Programmatic Configuration

```python
from othertales.deepresearch.enhanced_graph import create_enhanced_graph
from othertales.deepresearch.configuration import Configuration

# Configure the system
config = Configuration(
    # Domain mode
    agent_mode="legal",  # "legal", "tax", or "general"
    
    # Knowledge and verification
    use_knowledge_storage=True,
    verification_required=True,
    min_verification_confidence=0.7,
    
    # Search configuration
    search_api="tavily",
    number_of_queries=3,
    max_search_depth=2,
    
    # LLM models
    planner_provider="anthropic",
    planner_model="claude-3-7-sonnet-latest",
    writer_provider="anthropic",
    writer_model="claude-3-5-sonnet-latest",
    
    # Verification models
    verification_provider="anthropic",
    fact_check_model="claude-3-5-sonnet-latest",
    source_eval_model="claude-3-5-sonnet-latest",
    consistency_model="claude-3-5-sonnet-latest",
    legal_model="claude-3-sonnet-latest",
    tax_model="claude-3-sonnet-latest",
    
    # Storage configuration
    mongodb_uri="mongodb+srv://username:password@cluster.mongodb.net/deepresearch",
    neo4j_uri="neo4j://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="password"
)

# Create the enhanced graph
graph = create_enhanced_graph()

# Run the research
result = await graph.ainvoke({"topic": "Check Current Legislation on Derivative Works", "mode": "legal"})
```

## 📦 Search Tools

Available search tools:

* [Tavily API](https://tavily.com/) - General web search
* [Perplexity API](https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api) - General web search
* [Exa API](https://exa.ai/) - Powerful neural search for web content
* [ArXiv](https://arxiv.org/) - Academic papers
* [PubMed](https://pubmed.ncbi.nlm.nih.gov/) - Biomedical literature
* [Linkup API](https://www.linkup.so/) - General web search
* [DuckDuckGo API](https://duckduckgo.com/) - General web search
* [Google Search API/Scrapper](https://google.com/)
* [Direct URL Crawling](https://github.com/langchain-ai/othertales.deepresearch) - Extract content from specific URLs

## 🗄️ Knowledge Storage

The system uses two complementary storage systems:

### MongoDB Atlas Vector Store
- Stores document content and embeddings
- Enables semantic similarity search
- Tracks verification status and metadata

### Neo4j Knowledge Graph
- Maintains relationships between entities
- Enables complex graph queries
- Supports domain-specific relationship types

Document model includes:
- Unique identifier
- Content and title
- Source URL (if applicable)
- Domain classification (legal, tax, general)
- Verification status and details
- Embeddings for similarity search
- Relationships to other entities

## 🔍 Verification System

The multi-agent verification system ensures high accuracy:

### Verification Agents
1. **FactCheckAgent**: Verifies factual accuracy
2. **SourceEvaluationAgent**: Assesses source reliability
3. **ConsistencyAgent**: Checks for internal consistency
4. **LegalCorrectnessAgent**: Verifies Legal Compliance
5. **TaxComplianceAgent**: Verifies Accounting and Tax Compliance

### Verification Process
1. Document is processed for verification
2. Multiple independent agents verify different aspects
3. Only passes verification if all agents approve
4. Verification status and details are recorded
5. Verified documents are stored for future use

![deep-verification-workflow](/img/deep-verification.png)

## 🚀 Deployment

### Google Cloud Run Deployment

The system is ready for deployment on Google Cloud Run:

1. Build the Docker image:
```bash
docker build -t deep-research:latest .
```

2. Tag and push to Google Container Registry:
```bash
docker tag deep-research:latest gcr.io/[PROJECT-ID]/deep-research:latest
docker push gcr.io/[PROJECT-ID]/deep-research:latest
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy deep-research \
  --image gcr.io/[PROJECT-ID]/deep-research:latest \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars="MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/deepresearch,NEO4J_URI=neo4j+s://cluster-id.databases.neo4j.io,NEO4J_USERNAME=neo4j,NEO4J_PASSWORD=password" \
  --region us-central1
```

### Required Environment Variables

For proper deployment, set the following:
- `MONGODB_URI`: MongoDB Atlas connection string
- `NEO4J_URI`: Neo4j database URI
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`: LLM provider API keys
- `TAVILY_API_KEY` and/or other search API keys

## 📊 Testing Report Quality

Compare the quality of reports generated by different implementations:

```bash
# Test with default Anthropic models
python tests/run_test.py --all

# Test with OpenAI models and enhanced options
python tests/run_test.py --all \
  --supervisor-model "openai:o3" \
  --researcher-model "openai:o3" \
  --planner-provider "openai" \
  --planner-model "o3" \
  --writer-provider "openai" \
  --writer-model "o3" \
  --eval-model "openai:o3" \
  --search-api "tavily" \
  --agent-mode "legal"
```

The test results will be logged to LangSmith, allowing you to compare report quality with different configurations.

## 📋 Model Considerations

1. **Compatible Models**: You can use models supported with [the `init_chat_model()` API](https://python.langchain.com/docs/how_to/chat_models_universal_init/).

2. **Structured Output Support**: The workflow planner and writer models need to support structured outputs. Check whether structured outputs are supported by your chosen model [here](https://python.langchain.com/docs/integrations/chat/).

3. **Tool Calling Support**: The agent models need to support tool calling. Tests have been done with Claude 3.7, o3, o3-mini, and gpt4.1.

4. **Rate Limits**: Be aware of rate limits for models like Groq, which has token per minute (TPM) limits on the `on_demand` service tier.

## 🔄 Architecture

The enhanced system follows this workflow:

1. User submits research topic with domain mode
2. System generates initial plan with domain-specific guidance
3. For each research section:
   - Check existing verified knowledge first
   - If sufficient information exists, use it directly
   - If not, generate domain-optimized search queries
   - Process search results into structured sources
   - Verify sources using domain-appropriate verification agents
   - Store verified sources in knowledge base
   - Write section using only verified information
4. Compile verified sections into final report
5. Return final report with verification status and sources

## 📄 License

MIT.

Copyright © 2025 Adventures of the Persistently Impaired (...and Other Tales) Limited of 85 Great Portland Street, London W1W 7LT under exclusive license to Other Tales LLC of 8 The Green, Suite B, Dover DE 19901 United States.
