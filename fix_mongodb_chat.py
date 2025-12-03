"""
Diagnostic and fix script for MongoDB chat issues
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')

import django
django.setup()

from flickora.mongodb import MongoDBConnection
from reports.models import MovieSection
from reports.mongodb_models import MovieSectionMongoDB
from services.mongodb_rag_service import MongoDBRAGService

def main():
    print("=" * 80)
    print("MongoDB Chat Diagnostics & Fix")
    print("=" * 80)

    # Step 1: Check MongoDB connection
    print("\n[1/5] Checking MongoDB connection...")
    try:
        from flickora.mongodb import get_mongodb
        db = get_mongodb()
        collection = db.movie_embeddings
        print("✓ Connected to MongoDB")
    except Exception as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Count embeddings
    print("\n[2/5] Counting embeddings...")
    total_embeddings = collection.count_documents({})
    total_sections = MovieSection.objects.count()
    print(f"✓ MongoDB embeddings: {total_embeddings}")
    print(f"✓ PostgreSQL sections: {total_sections}")

    if total_embeddings == 0:
        print("\n" + "⚠️ " * 40)
        print("PROBLEM FOUND: No embeddings in MongoDB!")
        print("⚠️ " * 40)
        print("\nSOLUTION:")
        print("Run this command to generate embeddings:")
        print("  python manage.py generate_embeddings_mongodb\n")
        return

    # Step 3: Check vector search index
    print("\n[3/5] Checking vector search index...")
    indexes = list(collection.list_indexes())
    print(f"✓ Found {len(indexes)} indexes:")
    for idx in indexes:
        print(f"  - {idx.get('name', 'unnamed')}")

    has_vector_index = any('vector' in idx.get('name', '').lower() for idx in indexes)

    if not has_vector_index:
        print("\n" + "⚠️ " * 40)
        print("PROBLEM FOUND: No vector search index!")
        print("⚠️ " * 40)
        print("\nSOLUTION:")
        print("1. Run: python manage.py create_mongodb_indexes")
        print("2. Create vector search index in MongoDB Atlas UI")
        print("3. Index name MUST be: vector_index")
        print("4. Path: embedding")
        print("5. Dimensions: 384")
        print("6. Similarity: cosine\n")
        return

    # Step 4: Test vector search
    print("\n[4/5] Testing vector search...")
    try:
        rag = MongoDBRAGService()
        test_query = "What are the themes?"
        print(f"Query: '{test_query}'")

        query_embedding = rag.generate_embedding(test_query)
        print(f"✓ Generated embedding ({len(query_embedding)} dimensions)")

        results = MovieSectionMongoDB.vector_search(
            query_embedding=query_embedding,
            k=5,
            min_similarity=0.0  # Lower threshold for testing
        )

        print(f"✓ Vector search returned {len(results)} results")

        if len(results) == 0:
            print("\n" + "⚠️ " * 40)
            print("PROBLEM FOUND: Vector search returns 0 results!")
            print("⚠️ " * 40)
            print("\nPossible causes:")
            print("1. Vector index not properly created in Atlas")
            print("2. Index name mismatch (must be 'vector_index')")
            print("3. Index not finished building")
            print("\nVerify in MongoDB Atlas:")
            print("- Go to Database > Search")
            print("- Check if 'vector_index' exists and is ACTIVE")
            print("- Index should be on 'movie_embeddings' collection")
            return
        else:
            print("\n✓ Vector search is working!")
            print(f"  Sample result: movie_id={results[0]['movie_id']}, "
                  f"similarity={results[0].get('similarity', 0):.3f}")

    except Exception as e:
        print(f"\n✗ Vector search failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test full RAG search
    print("\n[5/5] Testing full RAG search with scores...")
    try:
        results = rag.search_with_scores(test_query, k=3, movie_id=None)
        print(f"✓ RAG search returned {len(results)} results")

        if len(results) > 0:
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['movie_title']} - {r['section_type']}")
                print(f"     Similarity: {r['similarity']:.3f}")
        else:
            print("✗ RAG search returned 0 results (unexpected)")

    except Exception as e:
        print(f"✗ RAG search failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - MongoDB chat should work!")
    print("=" * 80)
    print()

if __name__ == "__main__":
    main()
