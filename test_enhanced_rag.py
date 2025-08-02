#!/usr/bin/env python3
"""
Test Enhanced RAG System with Markdown Support
Demonstrates processing both JSON and Markdown files
"""
import sys
import os
sys.path.append('cognition_base')

from enhanced_rag_service import EnhancedRAGService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_enhanced_rag_system():
    """Test the enhanced RAG system with both JSON and Markdown files"""
    
    print("🧠 TESTING ENHANCED RAG SYSTEM WITH MARKDOWN SUPPORT")
    print("=" * 70)
    
    try:
        # Initialize enhanced RAG service
        print("📚 Initializing Enhanced RAG service...")
        rag_service = EnhancedRAGService(
            opensearch_host="localhost",
            opensearch_port=9200,
            index_name="enhanced_bayesian_rag"
        )
        
        # Load cognition data (both JSON and Markdown)
        print("📖 Loading cognition data (JSON + Markdown)...")
        documents = rag_service.load_cognition_data("cognition_base/cognition")
        
        if not documents:
            print("❌ No documents loaded! Make sure:")
            print("   1. OpenSearch is running: docker-compose up -d in cognition_base/")
            print("   2. Files exist in cognition_base/cognition/")
            return False
        
        # Analyze loaded documents
        json_docs = [d for d in documents if d.content_type == 'json_structured']
        md_docs = [d for d in documents if d.content_type == 'markdown']
        
        print(f"✅ Loaded {len(documents)} documents total:")
        print(f"   📄 JSON structured: {len(json_docs)}")
        print(f"   📝 Markdown files: {len(md_docs)}")
        
        # Show examples of each type
        if json_docs:
            print(f"\n📄 JSON Example: {json_docs[0].paper_key}")
            print(f"   Title: {json_docs[0].title}")
        
        if md_docs:
            print(f"\n📝 Markdown Example: {md_docs[0].paper_key}")
            print(f"   Title: {md_docs[0].title}")
            print(f"   Content preview: {md_docs[0].content[:100]}...")
        
        # Index the documents
        print(f"\n🔍 Indexing {len(documents)} documents...")
        success = rag_service.index_documents(documents)
        
        if not success:
            print("❌ Document indexing failed!")
            return False
        
        print("✅ Documents indexed successfully")
        
        # Test mixed queries (should find both JSON and Markdown content)
        mixed_queries = [
            "hierarchical models with PyMC code examples",
            "MCMC convergence diagnostics and R-hat",
            "Bayesian model comparison using WAIC",
            "PyMC sampling and posterior predictive checks",
            "group-specific parameters in hierarchical regression",
            "practical implementation of Bayesian models"
        ]
        
        print(f"\n🔬 TESTING MIXED CONTENT QUERIES:")
        print("=" * 50)
        
        for i, query in enumerate(mixed_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            results = rag_service.search_similar_patterns(query, k=3, similarity_threshold=0.4)
            
            if results:
                print(f"   📈 Found {len(results)} relevant results:")
                for j, result in enumerate(results, 1):
                    content_type_icon = "📄" if result['content_type'] == 'json_structured' else "📝"
                    print(f"      {j}. {content_type_icon} {result['title']}")
                    print(f"         Source: {result['paper_key']} ({result['content_type']})")
                    print(f"         Similarity: {result['score']:.3f}")
                    
                    # Show different information based on content type
                    if result['content_type'] == 'markdown':
                        print(f"         Has full content: {len(result.get('FULL_CONTENT', ''))} chars")
                    else:
                        print(f"         Structured insight: {result['DESIGN_INSIGHT'][:60]}...")
            else:
                print("   ⚠️  No results found above similarity threshold")
            print()
        
        print("✅ Enhanced RAG system testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced RAG system testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_enhanced_rag_system()
    
    if success:
        print(f"\n🎉 ENHANCED RAG SYSTEM READY!")
        print("\n📋 You can now:")
        print("✅ Drop JSON files in cognition_base/cognition/ (structured format)")
        print("✅ Drop Markdown files in cognition_base/cognition/ (PyMC notebooks, docs)")
        print("✅ Both will be automatically processed and indexed")
        print("✅ Agents can query both types of content seamlessly")
        
        print(f"\n📝 Markdown File Benefits:")
        print("• No need to convert to JSON format")
        print("• Preserves full notebook content")
        print("• Automatically extracts PyMC patterns")
        print("• Perfect for PyMC example notebooks")
        print("• Easy to add documentation")
        
        print(f"\n🚀 Next steps:")
        print("1. Add PyMC notebook .md files to cognition_base/cognition/")
        print("2. Start RAG API with enhanced service")
        print("3. Agents will automatically access both JSON and Markdown content")
        
    else:
        print(f"\n🔧 Setup needed:")
        print("1. Start OpenSearch: cd cognition_base && docker-compose up -d")
        print("2. Run this test again to verify functionality")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)