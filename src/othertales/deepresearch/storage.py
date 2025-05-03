"""
Document storage system for deep research agents with vector and graph storage.

This module provides a unified interface for storing and retrieving documents
using a combination of MongoDB Atlas Vector Search and Neo4j Graph Database.
The system stores both the raw content of documents as well as their vector 
embeddings for similarity search and their relationships in a knowledge graph.

Main components:
- DocumentStore: Core interface for document storage and retrieval
- MongoVectorStore: Implementation of vector storage using MongoDB Atlas
- Neo4jKnowledgeGraph: Implementation of knowledge graph using Neo4j
- VerifiedDocument: Data model for verified documents
"""

import os
import json
import uuid
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from enum import Enum

# External dependencies
import pymongo
from pymongo import MongoClient
from neo4j import GraphDatabase
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.embeddings import QdrantAtlasEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_mongodb.vectorstores.atlas import MongoDBAtlasVectorSearch

# Configure logging
import logging
logger = logging.getLogger(__name__)

# Default embedding model settings
DEFAULT_EMBEDDING_MODEL = "openai"
DEFAULT_EMBEDDING_DIMENSIONS = 1536

class VerificationStatus(str, Enum):
    """Status of document verification."""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    PARTIAL = "partial"


@dataclass
class VerifiedDocument:
    """A document that has been verified by the system."""
    id: str  # Unique identifier
    content: str  # Raw content of the document
    title: str  # Title or description of the document
    url: Optional[str] = None  # Source URL if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    embedding: Optional[List[float]] = None  # Vector embedding
    domain: Literal["legal", "tax", "general"] = "general"  # Domain classification
    verification_status: VerificationStatus = VerificationStatus.PENDING
    verification_agents: List[str] = field(default_factory=list)  # IDs of agents that verified this document
    verification_details: Dict[str, Any] = field(default_factory=dict)  # Details of verification process
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # Graph relationships
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    def to_langchain_document(self) -> Document:
        """Convert to a LangChain document."""
        return Document(
            page_content=self.content,
            metadata={
                "id": self.id,
                "title": self.title,
                "url": self.url,
                "domain": self.domain,
                "verification_status": self.verification_status.value,
                **self.metadata
            }
        )

    @classmethod
    def from_langchain_document(cls, doc: Document, domain: str = "general") -> "VerifiedDocument":
        """Create from a LangChain document."""
        metadata = doc.metadata.copy() if doc.metadata else {}
        doc_id = metadata.pop("id", str(uuid.uuid4()))
        title = metadata.pop("title", "Untitled Document")
        url = metadata.pop("url", None)
        
        return cls(
            id=doc_id,
            content=doc.page_content,
            title=title,
            url=url,
            metadata=metadata,
            domain=domain,
        )


class MongoVectorStore:
    """Vector store implementation using MongoDB Atlas."""

    def __init__(
        self, 
        connection_string: Optional[str] = None,
        database_name: str = "deep_research",
        collection_name: str = "documents",
        embedding_key: str = "embedding",
        index_name: str = "vector_index",
        embedding_model: Optional[Union[str, Embeddings]] = None,
    ):
        """Initialize the MongoDB vector store.
        
        Args:
            connection_string: MongoDB Atlas connection string. Defaults to MONGODB_URI env var.
            database_name: Name of the database to use.
            collection_name: Name of the collection to store documents in.
            embedding_key: Field name to store embeddings.
            index_name: Name of the vector index.
            embedding_model: Embedding model or name to use. Defaults to DEFAULT_EMBEDDING_MODEL.
        """
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        if not self.connection_string:
            raise ValueError("MongoDB connection string must be provided via argument or MONGODB_URI env var")
            
        self.database_name = database_name
        self.collection_name = collection_name
        self.embedding_key = embedding_key
        self.index_name = index_name
        
        # Initialize MongoDB client
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]
        
        # Setup embedding model
        self._setup_embeddings(embedding_model)
        
        # Initialize Vector Store
        self.vector_store = self._initialize_vector_store()
        
    def _setup_embeddings(self, embedding_model: Optional[Union[str, Embeddings]]):
        """Set up the embedding model."""
        if isinstance(embedding_model, Embeddings):
            self.embeddings = embedding_model
        else:
            # Use the provided model name or default
            model_name = embedding_model or DEFAULT_EMBEDDING_MODEL
            if model_name == "openai":
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
    
    def _initialize_vector_store(self) -> MongoDBAtlasVectorSearch:
        """Initialize the MongoDB Atlas Vector Search."""
        return MongoDBAtlasVectorSearch.from_connection_string(
            self.connection_string,
            self.collection_name,
            self.embeddings,
            index_name=self.index_name,
            embedding_key=self.embedding_key,
            db_name=self.database_name
        )
    
    def add_documents(self, documents: List[VerifiedDocument]) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
            
        Returns:
            List of document IDs.
        """
        # Convert to LangChain documents
        langchain_docs = [doc.to_langchain_document() for doc in documents]
        
        # Add to vector store
        ids = self.vector_store.add_documents(langchain_docs)
        
        # For each document, update with additional fields if it doesn't already exist
        for doc in documents:
            self.collection.update_one(
                {"_id": doc.id},
                {"$set": {
                    "content": doc.content,
                    "title": doc.title,
                    "url": doc.url,
                    "metadata": doc.metadata,
                    "domain": doc.domain,
                    "verification_status": doc.verification_status.value,
                    "verification_agents": doc.verification_agents,
                    "verification_details": doc.verification_details,
                    "relationships": doc.relationships,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at
                }},
                upsert=True
            )
        
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VerifiedDocument]:
        """Search for documents similar to the query.
        
        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional filter to apply to search results.
            
        Returns:
            List of similar documents.
        """
        # Create default filter for verified documents if not specified
        if filter is None:
            filter = {"verification_status": VerificationStatus.VERIFIED.value}
        elif isinstance(filter, dict) and "verification_status" not in filter:
            filter["verification_status"] = VerificationStatus.VERIFIED.value
            
        # Perform similarity search
        docs = self.vector_store.similarity_search(query, k=k, filter=filter)
        
        # Convert to VerifiedDocument objects
        return [
            VerifiedDocument(
                id=doc.metadata.get("id", ""),
                content=doc.page_content,
                title=doc.metadata.get("title", "Untitled"),
                url=doc.metadata.get("url"),
                metadata={k: v for k, v in doc.metadata.items() 
                         if k not in ["id", "title", "url", "domain", "verification_status"]},
                domain=doc.metadata.get("domain", "general"),
                verification_status=VerificationStatus(doc.metadata.get(
                    "verification_status", VerificationStatus.PENDING.value)),
            )
            for doc in docs
        ]
    
    def update_verification_status(
        self, 
        doc_id: str, 
        status: VerificationStatus,
        agent_id: str,
        details: Dict[str, Any]
    ) -> bool:
        """Update the verification status of a document.
        
        Args:
            doc_id: ID of the document to update.
            status: New verification status.
            agent_id: ID of the agent performing verification.
            details: Verification details.
            
        Returns:
            Whether the update was successful.
        """
        result = self.collection.update_one(
            {"_id": doc_id},
            {"$set": {
                "verification_status": status.value,
                "updated_at": datetime.datetime.now()
            },
            "$push": {
                "verification_agents": agent_id,
                "verification_details": details
            }}
        )
        return result.matched_count > 0


class Neo4jKnowledgeGraph:
    """Knowledge graph implementation using Neo4j."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize the Neo4j knowledge graph.
        
        Args:
            uri: Neo4j database URI. Defaults to NEO4J_URI env var.
            username: Neo4j username. Defaults to NEO4J_USERNAME env var.
            password: Neo4j password. Defaults to NEO4J_PASSWORD env var.
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.username = username or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.username, self.password]):
            raise ValueError("Neo4j connection parameters must be provided via arguments or environment variables")
            
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        # Create schema constraints for unique document IDs
        with self.driver.session() as session:
            # Create constraints if they don't exist
            session.run(
                """
                CREATE CONSTRAINT document_id_unique IF NOT EXISTS
                FOR (d:Document) REQUIRE d.id IS UNIQUE
                """
            )
    
    def add_document(self, document: VerifiedDocument) -> bool:
        """Add a document to the knowledge graph.
        
        Args:
            document: Document to add.
            
        Returns:
            Whether the operation was successful.
        """
        with self.driver.session() as session:
            # Create document node
            create_result = session.run(
                """
                MERGE (d:Document {id: $id})
                ON CREATE SET 
                    d.title = $title,
                    d.url = $url,
                    d.domain = $domain,
                    d.verification_status = $verification_status,
                    d.created_at = datetime(),
                    d.content_preview = left($content, 200)
                ON MATCH SET 
                    d.title = $title,
                    d.url = $url,
                    d.domain = $domain,
                    d.verification_status = $verification_status,
                    d.updated_at = datetime(),
                    d.content_preview = left($content, 200)
                RETURN d
                """,
                id=document.id,
                title=document.title,
                url=document.url or "",
                domain=document.domain,
                content=document.content,
                verification_status=document.verification_status.value
            )
            
            # Add relationships
            for relationship in document.relationships:
                session.run(
                    """
                    MATCH (d:Document {id: $doc_id})
                    MERGE (target:Entity {id: $target_id})
                    ON CREATE SET 
                        target.name = $target_name,
                        target.type = $target_type
                    MERGE (d)-[r:$relation_type]->(target)
                    SET r.confidence = $confidence
                    """,
                    doc_id=document.id,
                    target_id=relationship["target_id"],
                    target_name=relationship["target_name"],
                    target_type=relationship["target_type"],
                    relation_type=relationship["relation_type"],
                    confidence=relationship.get("confidence", 1.0)
                )
                
            # Check if document was created
            return create_result.single() is not None
    
    def get_related_documents(
        self, 
        entity_name: str, 
        relation_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get documents related to an entity.
        
        Args:
            entity_name: Name of the entity to find related documents for.
            relation_type: Optional specific relationship type to filter by.
            limit: Maximum number of results to return.
            
        Returns:
            List of related document metadata.
        """
        with self.driver.session() as session:
            if relation_type:
                # Query with specific relationship type
                result = session.run(
                    """
                    MATCH (d:Document)-[r:$relation_type]->(e:Entity)
                    WHERE e.name CONTAINS $entity_name
                    RETURN d.id, d.title, d.domain, d.verification_status, 
                           e.name as entity_name, type(r) as relation_type
                    LIMIT $limit
                    """,
                    entity_name=entity_name,
                    relation_type=relation_type,
                    limit=limit
                )
            else:
                # Query any relationship type
                result = session.run(
                    """
                    MATCH (d:Document)-[r]->(e:Entity)
                    WHERE e.name CONTAINS $entity_name
                    RETURN d.id, d.title, d.domain, d.verification_status, 
                           e.name as entity_name, type(r) as relation_type
                    LIMIT $limit
                    """,
                    entity_name=entity_name,
                    limit=limit
                )
                
            return [dict(record) for record in result]
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query against the knowledge graph.
        
        Args:
            query: Cypher query to execute.
            params: Parameters for the query.
            
        Returns:
            List of records from the query result.
        """
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
    
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()


class DocumentStore:
    """Unified interface for document storage and retrieval."""
    
    def __init__(
        self,
        mongodb_connection: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        database_name: str = "deep_research",
        collection_name: str = "documents",
        embedding_model: Optional[Union[str, Embeddings]] = None,
    ):
        """Initialize the document store.
        
        Args:
            mongodb_connection: MongoDB Atlas connection string.
            neo4j_uri: Neo4j database URI.
            neo4j_username: Neo4j username.
            neo4j_password: Neo4j password.
            database_name: Name of the MongoDB database.
            collection_name: Name of the MongoDB collection.
            embedding_model: Embedding model or name to use.
        """
        # Initialize MongoDB Vector Store
        self.vector_store = MongoVectorStore(
            connection_string=mongodb_connection,
            database_name=database_name,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        
        # Initialize Neo4j Knowledge Graph
        self.knowledge_graph = Neo4jKnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        logger.info("Document store initialized successfully.")
    
    def add_document(self, document: VerifiedDocument) -> str:
        """Add a document to the document store.
        
        Args:
            document: Document to add.
            
        Returns:
            ID of the added document.
        """
        # Add to vector store
        document_ids = self.vector_store.add_documents([document])
        document_id = document_ids[0] if document_ids else document.id
        
        # Add to knowledge graph
        self.knowledge_graph.add_document(document)
        
        return document_id
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        domain: Optional[str] = None,
        verification_status: VerificationStatus = VerificationStatus.VERIFIED
    ) -> List[VerifiedDocument]:
        """Search for documents similar to the query.
        
        Args:
            query: Query string.
            k: Number of results to return.
            domain: Optional domain filter.
            verification_status: Filter by verification status.
            
        Returns:
            List of similar documents.
        """
        # Create filter
        filter_dict = {"verification_status": verification_status.value}
        if domain:
            filter_dict["domain"] = domain
            
        # Perform search
        return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
    
    def get_related_documents(
        self, 
        entity_name: str, 
        relation_type: Optional[str] = None,
        limit: int = 10
    ) -> List[VerifiedDocument]:
        """Get documents related to an entity from the knowledge graph.
        
        Args:
            entity_name: Name of the entity to find related documents for.
            relation_type: Optional specific relationship type to filter by.
            limit: Maximum number of results to return.
            
        Returns:
            List of related documents.
        """
        # Get related document metadata from knowledge graph
        related_docs_meta = self.knowledge_graph.get_related_documents(
            entity_name, relation_type, limit
        )
        
        # Fetch full documents from MongoDB
        docs = []
        for doc_meta in related_docs_meta:
            doc_id = doc_meta.get("d.id")
            if doc_id:
                # Get document from MongoDB
                doc_data = self.vector_store.collection.find_one({"_id": doc_id})
                if doc_data:
                    docs.append(VerifiedDocument(
                        id=doc_id,
                        content=doc_data.get("content", ""),
                        title=doc_data.get("title", ""),
                        url=doc_data.get("url"),
                        metadata=doc_data.get("metadata", {}),
                        domain=doc_data.get("domain", "general"),
                        verification_status=VerificationStatus(doc_data.get(
                            "verification_status", VerificationStatus.PENDING.value)),
                        verification_agents=doc_data.get("verification_agents", []),
                        verification_details=doc_data.get("verification_details", {}),
                        relationships=doc_data.get("relationships", []),
                        created_at=doc_data.get("created_at", datetime.datetime.now()),
                        updated_at=doc_data.get("updated_at", datetime.datetime.now())
                    ))
        
        return docs
    
    def update_verification_status(
        self, 
        doc_id: str, 
        status: VerificationStatus,
        agent_id: str,
        details: Dict[str, Any]
    ) -> bool:
        """Update the verification status of a document.
        
        Args:
            doc_id: ID of the document to update.
            status: New verification status.
            agent_id: ID of the agent performing verification.
            details: Verification details.
            
        Returns:
            Whether the update was successful.
        """
        # Update in vector store
        success = self.vector_store.update_verification_status(
            doc_id, status, agent_id, details
        )
        
        # Update in knowledge graph
        if success:
            with self.knowledge_graph.driver.session() as session:
                session.run(
                    """
                    MATCH (d:Document {id: $id})
                    SET d.verification_status = $status,
                        d.updated_at = datetime()
                    """,
                    id=doc_id,
                    status=status.value
                )
        
        return success
    
    def close(self):
        """Close all connections."""
        self.knowledge_graph.close()
        self.vector_store.client.close()


# Factory function to create document store
def create_document_store(
    mongodb_connection: Optional[str] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    database_name: str = "deep_research",
    collection_name: str = "documents",
    embedding_model: Optional[Union[str, Embeddings]] = None,
) -> DocumentStore:
    """Create a document store instance.
    
    This function will create a DocumentStore using either provided connection
    parameters or environment variables.
    
    Args:
        mongodb_connection: MongoDB Atlas connection string.
        neo4j_uri: Neo4j database URI.
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        database_name: Name of the MongoDB database.
        collection_name: Name of the MongoDB collection.
        embedding_model: Embedding model or name to use.
        
    Returns:
        DocumentStore instance.
    """
    try:
        # Use provided parameters or environment variables
        mongodb_conn = mongodb_connection or os.getenv("MONGODB_URI")
        neo4j_uri_val = neo4j_uri or os.getenv("NEO4J_URI")
        neo4j_user = neo4j_username or os.getenv("NEO4J_USERNAME")
        neo4j_pass = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        # Check if required variables are set
        missing_vars = []
        if not mongodb_conn:
            missing_vars.append("MONGODB_URI")
        if not neo4j_uri_val:
            missing_vars.append("NEO4J_URI")
        if not neo4j_user:
            missing_vars.append("NEO4J_USERNAME")
        if not neo4j_pass:
            missing_vars.append("NEO4J_PASSWORD")
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Create document store
        return DocumentStore(
            mongodb_connection=mongodb_conn,
            neo4j_uri=neo4j_uri_val,
            neo4j_username=neo4j_user,
            neo4j_password=neo4j_pass,
            database_name=database_name,
            collection_name=collection_name,
            embedding_model=embedding_model
        )
    
    except Exception as e:
        logger.error(f"Failed to create document store: {e}")
        raise