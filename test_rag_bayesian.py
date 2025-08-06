#!/usr/bin/env python3
"""
Test RAG System with Bayesian Statistics Content
"""
import sys
import os
sys.path.append('cognition_base')

from rag_service import OpenSearchRAGService
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_bayesian_rag_system():
    """Test the RAG system with Bayesian modeling queries"""
    
    print("🧠 TESTING RAG SYSTEM WITH BAYESIAN CONTENT")
    print("=" * 60)
    
    try:
        # Initialize RAG service
        print("📚 Initializing RAG service...")
        rag_service = OpenSearchRAGService(
            opensearch_host="localhost",
            opensearch_port=9200,
            index_name="bayesian_cognition_rag"
        )
        
        # Load cognition data (including our new Bayesian files)
        print("📖 Loading Bayesian cognition data...")
        documents = rag_service.load_cognition_data("cognition_base/cognition")
        
        if not documents:
            print("❌ No documents loaded! Make sure:")
            print("   1. OpenSearch is running: docker-compose up -d in cognition_base/")
            print("   2. Bayesian JSON files exist in cognition_base/cognition/")
            return False
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Index the documents
        print("🔍 Indexing documents...")
        success = rag_service.index_documents(documents)
        
        if not success:
            print("❌ Document indexing failed!")
            return False
        
        print("✅ Documents indexed successfully")
        
        # Get statistics
        stats = rag_service.get_stats()
        print(f"\n📊 INDEX STATISTICS:")
        print(f"   Total Documents: {stats.get('total_documents', 0)}")
        print(f"   Unique Papers: {stats.get('unique_papers', 0)}")
        print(f"   Index Name: {stats.get('index_name', 'unknown')}")
        
        # Test Bayesian-specific queries
        bayesian_queries = [
            "hierarchical data with multiple groups requiring partial pooling",
            "MCMC convergence issues with divergent transitions",
            "model comparison using WAIC and LOO cross-validation", 
            "non-centered parameterization for hierarchical models",
            "marginal likelihood estimation and Bayes factors",
            "mixture models for multimodal distributions",
            "robust regression with heavy-tailed likelihoods",
            "posterior predictive checks for model validation"
        ]
        
        print(f"\n🔬 TESTING BAYESIAN QUERIES:")
        print("=" * 40)
        
        for i, query in enumerate(bayesian_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            results = rag_service.search_similar_patterns(query, k=3, similarity_threshold=0.5)
            
            if results:
                print(f"   📈 Found {len(results)} relevant results:")
                for j, result in enumerate(results, 1):
                    print(f"      {j}. Paper: {result['paper_key']}")
                    print(f"         Similarity: {result['score']:.3f}")
                    print(f"         Insight: {result['DESIGN_INSIGHT'][:80]}...")
            else:
                print("   ⚠️  No results found above similarity threshold")
            print()
        
        print("✅ RAG system testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_bayesian_rag_system()
    
    if success:
        print(f"\n🎉 RAG SYSTEM READY FOR BAYESIAN MODELING!")
        print("\n📋 Next steps:")
        print("1. Start RAG API: cd cognition_base && python rag_api.py")
        print("2. The autonomous agents can now query Bayesian knowledge")
        print("3. Add more Bayesian papers following the same JSON format")
        print("4. Test agent integration with Bayesian queries")
    else:
        print(f"\n🔧 RAG system needs setup:")
        print("1. Start OpenSearch: cd cognition_base && docker-compose up -d")
        print("2. Ensure Bayesian JSON files are in cognition_base/cognition/")
        print("3. Run this test again to verify functionality")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)