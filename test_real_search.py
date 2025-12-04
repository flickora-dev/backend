"""
Test the actual RAG search with a real query
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')

import django
django.setup()

from services.mongodb_rag_service import MongoDBRAGService

def main():
    print("=" * 80)
    print("Testing Real MongoDB RAG Search")
    print("=" * 80)

    rag = MongoDBRAGService()

    # Test query
    query = "What are the main themes of the movie?"
    print(f"\nQuery: '{query}'")

    # Generate embedding
    print("\n[1/3] Generating query embedding...")
    try:
        embedding = rag.generate_embedding(query)
        print(f"[OK] Generated embedding: {len(embedding)} dimensions")
        print(f"     First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"[ERROR] Failed to generate embedding: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try direct MongoDB search
    print("\n[2/3] Calling MongoDB vector search...")
    try:
        from reports.mongodb_models import MovieSectionMongoDB

        results = MovieSectionMongoDB.vector_search(
            query_embedding=embedding,
            k=5,
            min_similarity=0.0
        )

        print(f"[OK] Direct MongoDB search returned: {len(results)} results")
        if len(results) > 0:
            print("\nFirst result:")
            print(f"  - movie_id: {results[0].get('movie_id')}")
            print(f"  - section_type: {results[0].get('section_type')}")
            print(f"  - similarity: {results[0].get('similarity', 0):.4f}")
    except Exception as e:
        print(f"[ERROR] Direct MongoDB search failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try full RAG search
    print("\n[3/3] Calling full RAG search_with_priority...")
    try:
        results = rag.search_with_priority(query, k=5)

        print(f"[OK] RAG search_with_priority returned: {len(results)} results")
        if len(results) > 0:
            print("\nFirst result:")
            print(f"  - movie_id: {results[0].get('movie_id')}")
            print(f"  - section_type: {results[0].get('section_type')}")
            print(f"  - similarity: {results[0].get('similarity', 0):.4f}")
            print(f"  - weighted_score: {results[0].get('weighted_score', 0):.4f}")
        else:
            print("[ERROR] RAG returned 0 results!")
    except Exception as e:
        print(f"[ERROR] RAG search failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
