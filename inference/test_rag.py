"""
Test script for multimodal retrieval in the inference pipeline
"""
import json
from rag.multimodal_retriever import MultimodalRetriever
from inference_pipeline import RAG_Orchestrator

def test_multimodal_retriever():
    """
    Test the MultimodalRetriever class
    """
    print("Testing MultimodalRetriever...")
    
    # Test queries
    queries = [
        "Explain the architecture diagram in the document",
        "What are the main components of the system?",
        "How does the data flow through the pipeline?"
    ]
    
    image_query = "technical architecture diagram with components"
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print(f"Image query: '{image_query}'")
        
        # Create retriever and get results
        retriever = MultimodalRetriever(query=query, image_query=image_query)
        results = retriever.retrieve_top_k(k_text=3, k_images=2)
        
        # Print text results
        print(f"\nFound {len(results['text_results'])} text results:")
        for i, result in enumerate(results['text_results']):
            print(f"  {i+1}. Score: {result['score']}")
            print(f"     Source: {result['source']}")
            print(f"     Content: {result['content'][:100]}...")
        
        # Print image results
        print(f"\nFound {len(results['image_results'])} image results:")
        for i, result in enumerate(results['image_results']):
            print(f"  {i+1}. Score: {result['score']}")
            print(f"     Image ID: {result['image_id']}")
            print(f"     Caption: {result['caption']}")
            print(f"     Source: {result['source']}")
        
        print("-" * 80)

def test_rag_orchestrator():
    """
    Test the RAG_Orchestrator class
    """
    print("Testing RAG_Orchestrator...")
    
    # Create orchestrator
    orchestrator = RAG_Orchestrator()
    
    # Test a query
    query = "Explain how the multimodal RAG system works"
    image_query = "system architecture diagram"
    
    print(f"\nQuery: '{query}'")
    print(f"Image query: '{image_query}'")
    
    # Get results
    results = orchestrator.retrieve(
        query=query, 
        image_query=image_query,
        k_text=3,
        k_images=2
    )
    
    # Print generated prompt
    print("\nGenerated prompt:")
    print("-" * 80)
    print(results["prompt"])
    print("-" * 80)
    
    # Print retrieval stats
    print(f"\nRetrieved {len(results['text_results'])} text chunks and {len(results['image_results'])} images")

if __name__ == "__main__":
    print("Running multimodal retrieval tests...")
    
    # Test the MultimodalRetriever
    test_multimodal_retriever()
    
    # Test the RAG_Orchestrator
    test_rag_orchestrator()
    
    print("\nTests completed")