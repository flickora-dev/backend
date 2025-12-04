"""
Simple MongoDB vector search test without emojis
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')

import django
django.setup()

from flickora.mongodb import get_mongodb
import numpy as np

def main():
    print("=" * 80)
    print("MongoDB Vector Search Diagnostic")
    print("=" * 80)

    db = get_mongodb()
    collection = db.movie_embeddings

    # Count documents
    total = collection.count_documents({})
    print(f"\n[1/3] Total documents in collection: {total}")

    if total == 0:
        print("[ERROR] No documents found! You need to generate embeddings first.")
        print("Run: python manage.py generate_embeddings_mongodb")
        return

    # Get a sample document
    sample = collection.find_one()
    if sample:
        print(f"\n[2/3] Sample document:")
        print(f"  - movie_id: {sample.get('movie_id')}")
        print(f"  - section_type: {sample.get('section_type')}")
        print(f"  - embedding length: {len(sample.get('embedding', []))}")

    # Test vector search
    print(f"\n[3/3] Testing vector search...")
    query_vector = np.random.randn(384).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5,
            }
        },
        {
            "$addFields": {
                "similarity": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    print("\nExecuting $vectorSearch aggregation pipeline...")
    try:
        results = list(collection.aggregate(pipeline))
        print(f"\n[SUCCESS] Vector search returned {len(results)} results")

        if len(results) > 0:
            print("\nTop results:")
            for i, r in enumerate(results[:3], 1):
                print(f"  {i}. movie_id={r.get('movie_id')}, similarity={r.get('similarity', 0):.4f}")
        else:
            print("\n[WARNING] Search succeeded but returned 0 results")
            print("This means:")
            print("  1. Vector index 'vector_index' exists")
            print("  2. But it may not be properly configured or built")
            print("\nCheck in MongoDB Atlas:")
            print("  - Database > Search tab")
            print("  - Verify index 'vector_index' status is ACTIVE (not Building)")

    except Exception as e:
        error_str = str(e)
        print(f"\n[FAILED] Vector search error:")
        print(f"  {error_str}")

        if "vector_index" in error_str.lower() or "index" in error_str.lower():
            print("\n" + "=" * 80)
            print("PROBLEM: Vector index 'vector_index' not found or misconfigured!")
            print("=" * 80)
            print("\nSOLUTION:")
            print("1. Go to MongoDB Atlas > Database > Search")
            print("2. Create a new Vector Search index with:")
            print("   - Index Name: vector_index")
            print("   - Collection: movie_embeddings")
            print("   - Definition:")
            print("""
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
            """)
        else:
            print("\nFull error traceback:")
            import traceback
            traceback.print_exc()

    # Check indexes
    print("\n" + "=" * 80)
    print("Checking collection indexes...")
    print("=" * 80)

    indexes = list(collection.list_indexes())
    print(f"\nFound {len(indexes)} standard indexes:")
    for idx in indexes:
        print(f"  - {idx.get('name', 'unnamed')}: {idx.get('key', {})}")

    print("\nNote: Vector Search indexes created in Atlas UI")
    print("      may not appear in list_indexes() - this is normal.")

    print("\n" + "=" * 80)
    print("Diagnostic Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
