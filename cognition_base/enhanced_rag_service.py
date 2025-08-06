#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG Service with Markdown Support
Supports both JSON structured content and Markdown files (like PyMC notebooks)
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGDocument:
    """Enhanced RAG Document Data Structure"""
    id: str
    paper_key: str  # Filename
    title: str
    content: str
    content_type: str  # 'json_structured' or 'markdown'
    design_insight: str
    experimental_trigger_patterns: str
    background: str
    algorithmic_innovation: str
    implementation_guidance: str
    design_ai_instructions: str
    embedding: Optional[List[float]] = None

class MarkdownProcessor:
    """Process Markdown files for RAG indexing"""
    
    @staticmethod
    def extract_sections(markdown_content: str) -> Dict[str, str]:
        """Extract sections from markdown content"""
        sections = {
            'title': '',
            'content': markdown_content,
            'code_blocks': [],
            'headers': []
        }
        
        # Extract title (first # header)
        title_match = re.search(r'^#\s+(.+?)$', markdown_content, re.MULTILINE)
        if title_match:
            sections['title'] = title_match.group(1).strip()
        
        # Extract all headers
        headers = re.findall(r'^#{1,6}\s+(.+?)$', markdown_content, re.MULTILINE)
        sections['headers'] = headers
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:python|pymc)?\n(.*?)\n```', markdown_content, re.DOTALL)
        sections['code_blocks'] = code_blocks
        
        return sections
    
    @staticmethod
    def markdown_to_rag_format(filename: str, markdown_content: str) -> Dict[str, str]:
        """Convert markdown content to RAG format"""
        sections = MarkdownProcessor.extract_sections(markdown_content)
        
        # Extract PyMC-specific patterns
        pymc_patterns = []
        if 'pm.Model' in markdown_content:
            pymc_patterns.append("PyMC probabilistic programming")
        if 'pm.sample' in markdown_content:
            pymc_patterns.append("MCMC sampling")
        if 'pm.Normal' in markdown_content or 'pm.HalfNormal' in markdown_content:
            pymc_patterns.append("Bayesian parameter estimation")
        if 'az.summary' in markdown_content:
            pymc_patterns.append("MCMC diagnostics and convergence")
        if 'az.plot' in markdown_content:
            pymc_patterns.append("Bayesian visualization")
        
        # Create structured content
        return {
            'DESIGN_INSIGHT': f"### {sections['title'] or filename}",
            'EXPERIMENTAL_TRIGGER_PATTERNS': f"**PyMC_Usage_Patterns**:\n" + 
                                           "\n".join([f"- {pattern}" for pattern in pymc_patterns]) +
                                           f"\n**Content_Type**: Practical implementation guide with code examples\n" +
                                           f"**Application_Domain**: {', '.join(sections['headers'][:3])}",
            'BACKGROUND': f"**Source**: {filename}\n**Content_Type**: PyMC notebook/documentation\n" +
                         f"**Headers**: {', '.join(sections['headers'])}",
            'ALGORITHMIC_INNOVATION': f"**Implementation_Examples**:\n{len(sections['code_blocks'])} code blocks with PyMC implementations\n" +
                                    "**Methodological_Approach**: Practical Bayesian modeling examples",
            'IMPLEMENTATION_GUIDANCE': f"**PyMC_Code_Examples**:\nContains {len(sections['code_blocks'])} code examples\n" +
                                     "**Practical_Application**: Step-by-step implementation guidance\n" +
                                     f"**Full_Content**: {markdown_content[:500]}...",
            'DESIGN_AI_INSTRUCTIONS': "**Agent_Usage**: Reference for practical PyMC implementation patterns\n" +
                                    "**Code_Extraction**: Contains executable code examples\n" +
                                    "**Learning_Resource**: Use for implementation guidance and best practices"
        }

class LocalEmbeddingClient:
    """Local SentenceTransformer Embedding Client"""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.model_name = model_name
        logger.info(f"Successfully loaded model {model_name} (using CPU)")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding vector dimension: {self.embedding_dim}")
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        
        try:
            processed_texts = [f"query: {text}" if not text.startswith("query:") else text for text in texts]
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(batch_size, len(batch_texts))
                )
                all_embeddings.extend(batch_embeddings.tolist())
                logger.debug(f"Processed {min(i + batch_size, len(processed_texts))}/{len(processed_texts)} texts")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return [[0.0] * self.embedding_dim] * len(texts)
    
    def get_single_embedding(self, text: str) -> List[float]:
        try:
            processed_text = f"query: {text}" if not text.startswith("query:") else text
            embedding = self.model.encode(processed_text, convert_to_numpy=True, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting single embedding: {e}")
            return [0.0] * self.embedding_dim

class EnhancedRAGService:
    """Enhanced RAG Service supporting both JSON and Markdown files"""
    
    def __init__(
        self,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        index_name: str = "enhanced_cognition_rag",
        model_name: str = "intfloat/e5-base-v2"
    ):
        self.index_name = index_name
        
        # Initialize OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        # Initialize local embedding client
        logger.info("Initializing local embedding client")
        self.embedding_client = LocalEmbeddingClient(model_name=model_name)
        
        # Create the index
        self._create_index()
    
    def _create_index(self):
        """Creates the enhanced OpenSearch index"""
        embedding_dim = self.embedding_client.embedding_dim
        
        index_body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "paper_key": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "content_type": {"type": "keyword"},
                    "design_insight": {"type": "text"},
                    "experimental_trigger_patterns": {"type": "text"},
                    "background": {"type": "text"},
                    "algorithmic_innovation": {"type": "text"},
                    "implementation_guidance": {"type": "text"},
                    "design_ai_instructions": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 24}
                        }
                    }
                }
            }
        }
        
        # Delete existing index if it exists
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted existing index: {self.index_name}")
        
        # Create new index
        self.client.indices.create(index=self.index_name, body=index_body)
        logger.info(f"Created enhanced index: {self.index_name} (embedding dimension: {embedding_dim})")
    
    def _generate_document_id(self, paper_key: str, content_snippet: str) -> str:
        """Generates the document ID"""
        content = f"{paper_key}_{content_snippet[:100]}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _extract_filename(self, filepath: str) -> str:
        """Extract file name (without extension) from file path"""
        return Path(filepath).stem
    
    def load_cognition_data(self, data_dir: str = "cognition") -> List[RAGDocument]:
        """
        Loads both JSON and Markdown files from the cognition data directory
        
        Args:
            data_dir: Path to the data directory
            
        Returns:
            List of RAGDocument objects
        """
        documents = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return documents
        
        # Process JSON files (existing format)
        for json_file in data_path.glob("*.json"):
            documents.extend(self._process_json_file(json_file))
        
        # Process Markdown files (new format)
        for md_file in data_path.glob("*.md"):
            documents.extend(self._process_markdown_file(md_file))
        
        logger.info(f"Loaded {len(documents)} documents total")
        
        # Generate embeddings
        if documents:
            logger.info("Generating embeddings for all documents...")
            texts = [doc.experimental_trigger_patterns for doc in documents]
            embeddings = self.embedding_client.get_embeddings(texts)
            
            for i, doc in enumerate(documents):
                if i < len(embeddings):
                    doc.embedding = embeddings[i]
                else:
                    doc.embedding = [0.0] * self.embedding_client.embedding_dim
            
            logger.info(f"Generated {len(embeddings)} embedding vectors")
        
        return documents
    
    def _process_json_file(self, json_file: Path) -> List[RAGDocument]:
        """Process JSON file in original format"""
        documents = []
        try:
            logger.info(f"Processing JSON file: {json_file}")
            paper_key = self._extract_filename(str(json_file))
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                if isinstance(item, dict) and "EXPERIMENTAL_TRIGGER_PATTERNS" in item:
                    trigger_patterns = item.get("EXPERIMENTAL_TRIGGER_PATTERNS", "").strip()
                    if not trigger_patterns:
                        logger.warning(f"Skipping empty EXPERIMENTAL_TRIGGER_PATTERNS, file: {paper_key}")
                        continue
                    
                    doc_id = self._generate_document_id(paper_key, item.get("DESIGN_INSIGHT", ""))
                    
                    doc = RAGDocument(
                        id=doc_id,
                        paper_key=paper_key,
                        title=item.get("DESIGN_INSIGHT", "").replace("###", "").strip(),
                        content="",  # JSON doesn't have raw content
                        content_type="json_structured",
                        design_insight=item.get("DESIGN_INSIGHT", ""),
                        experimental_trigger_patterns=trigger_patterns,
                        background=item.get("BACKGROUND", ""),
                        algorithmic_innovation=item.get("ALGORITHMIC_INNOVATION", ""),
                        implementation_guidance=item.get("IMPLEMENTATION_GUIDANCE", ""),
                        design_ai_instructions=item.get("DESIGN_AI_INSTRUCTIONS", "")
                    )
                    
                    documents.append(doc)
        
        except Exception as e:
            logger.error(f"Error processing JSON file {json_file}: {e}")
        
        return documents
    
    def _process_markdown_file(self, md_file: Path) -> List[RAGDocument]:
        """Process Markdown file (like PyMC notebooks)"""
        documents = []
        try:
            logger.info(f"Processing Markdown file: {md_file}")
            paper_key = self._extract_filename(str(md_file))
            
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Convert markdown to structured format
            rag_format = MarkdownProcessor.markdown_to_rag_format(paper_key, markdown_content)
            
            doc_id = self._generate_document_id(paper_key, rag_format["DESIGN_INSIGHT"])
            
            doc = RAGDocument(
                id=doc_id,
                paper_key=paper_key,
                title=MarkdownProcessor.extract_sections(markdown_content)['title'] or paper_key,
                content=markdown_content,
                content_type="markdown",
                design_insight=rag_format["DESIGN_INSIGHT"],
                experimental_trigger_patterns=rag_format["EXPERIMENTAL_TRIGGER_PATTERNS"],
                background=rag_format["BACKGROUND"],
                algorithmic_innovation=rag_format["ALGORITHMIC_INNOVATION"],
                implementation_guidance=rag_format["IMPLEMENTATION_GUIDANCE"],
                design_ai_instructions=rag_format["DESIGN_AI_INSTRUCTIONS"]
            )
            
            documents.append(doc)
            
        except Exception as e:
            logger.error(f"Error processing Markdown file {md_file}: {e}")
        
        return documents
    
    def index_documents(self, documents: List[RAGDocument]) -> bool:
        """Index documents into OpenSearch with enhanced fields"""
        try:
            for doc in documents:
                doc_body = {
                    "paper_key": doc.paper_key,
                    "title": doc.title,
                    "content": doc.content,
                    "content_type": doc.content_type,
                    "design_insight": doc.design_insight,
                    "experimental_trigger_patterns": doc.experimental_trigger_patterns,
                    "background": doc.background,
                    "algorithmic_innovation": doc.algorithmic_innovation,
                    "implementation_guidance": doc.implementation_guidance,
                    "design_ai_instructions": doc.design_ai_instructions,
                    "embedding": doc.embedding
                }
                
                self.client.index(index=self.index_name, id=doc.id, body=doc_body)
            
            # Refresh the index
            self.client.indices.refresh(index=self.index_name)
            logger.info(f"Successfully indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    def search_similar_patterns(self, query: str, k: int = 5, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Search for similar patterns with enhanced results"""
        query_embedding = self.embedding_client.get_single_embedding(query)
        
        search_body = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
        
        try:
            response = self.client.search(index=self.index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                score = max(0.0, hit['_score'] - 1.0)  # Normalize
                
                if score >= similarity_threshold:
                    result = {
                        "id": hit['_id'],
                        "score": score,
                        "paper_key": hit['_source']['paper_key'],
                        "title": hit['_source']['title'],
                        "content_type": hit['_source']['content_type'],
                        "DESIGN_INSIGHT": hit['_source']['design_insight'],
                        "EXPERIMENTAL_TRIGGER_PATTERNS": hit['_source']['experimental_trigger_patterns'],
                        "BACKGROUND": hit['_source']['background'],
                        "ALGORITHMIC_INNOVATION": hit['_source']['algorithmic_innovation'],
                        "IMPLEMENTATION_GUIDANCE": hit['_source']['implementation_guidance'],
                        "DESIGN_AI_INSTRUCTIONS": hit['_source']['design_ai_instructions']
                    }
                    
                    # Add full content for markdown files
                    if hit['_source']['content_type'] == 'markdown':
                        result["FULL_CONTENT"] = hit['_source']['content']
                    
                    results.append(result)
            
            logger.info(f"Query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

def main():
    """Test the enhanced RAG service"""
    print("🧠 TESTING ENHANCED RAG SERVICE WITH MARKDOWN SUPPORT")
    print("=" * 70)
    
    # Initialize enhanced RAG service
    rag_service = EnhancedRAGService()
    
    # Load data (both JSON and Markdown)
    documents = rag_service.load_cognition_data("cognition")
    
    if not documents:
        print("❌ No documents loaded")
        return
    
    # Index documents
    success = rag_service.index_documents(documents)
    if not success:
        print("❌ Indexing failed")
        return
    
    # Print statistics
    json_docs = len([d for d in documents if d.content_type == 'json_structured'])
    md_docs = len([d for d in documents if d.content_type == 'markdown'])
    
    print(f"\n📊 ENHANCED INDEX STATISTICS:")
    print(f"   Total Documents: {len(documents)}")
    print(f"   JSON Structured: {json_docs}")
    print(f"   Markdown Files: {md_docs}")
    
    # Test queries
    test_queries = [
        "PyMC hierarchical modeling examples",
        "MCMC sampling convergence diagnostics",
        "Bayesian model implementation code",
        "posterior predictive checks"
    ]
    
    print(f"\n🔍 TESTING ENHANCED QUERIES:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag_service.search_similar_patterns(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']} ({result['content_type']})")
            print(f"     Similarity: {result['score']:.3f}")
            print(f"     Source: {result['paper_key']}")

if __name__ == "__main__":
    main()