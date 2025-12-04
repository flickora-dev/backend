"""
Direct test of MongoDB vector search to diagnose the issue
"""
import os
import sys
from pathlib import Path
import pprint

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')

import django
django.setup()

from flickora.mongodb import get_mongodb
import numpy as np

def test_vector_search():
    print("=" * 80)
    print("Direct Vector Search Test")
    print("=" * 80)

    db = get_mongodb()
    collection = db.movie_embeddings

    # Count documents
    total = collection.count_documents({})
    print(f"\n✓ Total documents in collection: {total}")

    if total == 0:
        print("✗ No documents found! Generate embeddings first.")
        return

    # Get one document to see structure
    sample = collection.find_one()
    print(f"\n✓ Sample document structure:")
    print(f"  - movie_id: {sample.get('movie_id')}")
    print(f"  - section_type: {sample.get('section_type')}")
    print(f"  - embedding length: {len(sample.get('embedding', []))}")

    # Create a test query vector
    # STRATEGY: Use a REAL embedding from the DB to ensure we get a match
    print(f"\n[TEST] Testing with REAL embedding from document {sample.get('_id')}...")
    query_vector = sample.get('embedding')
    
    if not query_vector or not isinstance(query_vector, list):
        print("✗ Sample document has invalid embedding!")
        return

    # Test 1: Try vector search with current code (Post-filtering)
    print(f"\n[TEST 1] Testing with Post-Filtering (Current App Logic)...")
    
    # Use the movie_id from the sample
    target_movie_id = sample.get('movie_id')
    print(f"Target Movie ID: {target_movie_id}")

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 100, # Get more results to increase chance of finding our movie
            }
        },
        {
            "$addFields": {
                "similarity": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$match": {"movie_id": target_movie_id}
        },
        {
            "$limit": 5
        }
    ]

    print("\nRunning pipeline with $match stage...")
    try:
        results = list(collection.aggregate(pipeline))
        print(f"✓ Found {len(results)} results")
        if len(results) == 0:
             print("⚠️  0 results with post-filtering! This confirms the issue.")
             print("   The movie was not found in the top 100 global matches.")
        else:
             print("✓ Post-filtering worked (target movie was in top 100).")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 2: Try vector search with Pre-filtering (The Fix)
    print(f"\n[TEST 2] Testing with Pre-Filtering (Requires Index Update)...")
    
    pipeline_pre = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "filter": {"movie_id": target_movie_id},
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

    print("\nRunning pipeline with 'filter' option...")
    try:
        results = list(collection.aggregate(pipeline_pre))
        print(f"✓ Found {len(results)} results")
        if len(results) > 0:
            print("✓ Pre-filtering works! (Index likely has filter field defined)")
        else:
            print("⚠️  0 results with pre-filtering. Index likely needs update.")
    except Exception as e:
        print(f"✗ Failed: {e}")
        print("  This likely means the index definition is missing the 'filter' field for movie_id.")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_vector_search()
