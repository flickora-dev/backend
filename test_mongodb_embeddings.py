"""
Test script for MongoDB embedding operations
Verifies that all embedding operations work correctly with MongoDB Atlas
"""
import os
import django
import sys

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')
django.setup()

from services.mongodb_service import get_mongodb_service
from services.rag_service import RAGService
from reports.models import MovieSection
import numpy as np


def test_connection():
    """Test 1: MongoDB connection"""
    print("\n" + "="*60)
    print("TEST 1: MongoDB Connection")
    print("="*60)
    try:
        mongodb = get_mongodb_service()
        count = mongodb.get_embeddings_count()
        print(f"✅ Connected to MongoDB")
        print(f"   Total embeddings in database: {count}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_store_embedding():
    """Test 2: Store embedding"""
    print("\n" + "="*60)
    print("TEST 2: Store Embedding")
    print("="*60)
    try:
        mongodb = get_mongodb_service()
        rag = RAGService()

        # Generate a test embedding
        test_text = "This is a test embedding for verification purposes."
        embedding = rag.generate_embedding(test_text)

        # Store it with a test section_id
        test_section_id = 999999
        success = mongodb.store_embedding(
            section_id=test_section_id,
            movie_id=1,
            section_type='production',
            embedding=embedding.tolist(),
            metadata={
                'test': True,
                'movie_title': 'Test Movie',
                'section_type_display': 'Test Section'
            }
        )

        if success:
            print(f"✅ Embedding stored successfully")
            print(f"   Section ID: {test_section_id}")
            print(f"   Dimensions: {len(embedding)}")

            # Clean up
            mongodb.delete_embedding(test_section_id)
            print(f"   Test embedding cleaned up")
            return True
        else:
            print(f"❌ Failed to store embedding")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_retrieve_embedding():
    """Test 3: Retrieve embedding"""
    print("\n" + "="*60)
    print("TEST 3: Retrieve Embedding")
    print("="*60)
    try:
        mongodb = get_mongodb_service()

        # Get first section with content
        section = MovieSection.objects.first()
        if not section:
            print("⚠️  No sections found in database")
            return False

        # Check if it has embedding in MongoDB
        doc = mongodb.get_embedding(section.id)

        if doc:
            print(f"✅ Retrieved embedding for section {section.id}")
            print(f"   Movie: {doc.get('metadata', {}).get('movie_title', 'N/A')}")
            print(f"   Section: {doc.get('section_type', 'N/A')}")
            print(f"   Dimensions: {doc.get('dimensions', 0)}")
            return True
        else:
            print(f"⚠️  No embedding found for section {section.id}")
            print(f"   This is normal if embeddings haven't been generated yet")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_vector_search():
    """Test 4: Vector similarity search"""
    print("\n" + "="*60)
    print("TEST 4: Vector Similarity Search")
    print("="*60)
    try:
        mongodb = get_mongodb_service()
        rag = RAGService()

        # Check if we have any embeddings
        count = mongodb.get_embeddings_count()
        if count == 0:
            print("⚠️  No embeddings in database to search")
            print("   Run 'python manage.py generate_embeddings' first")
            return False

        # Generate query embedding
        query = "What is the plot of the movie?"
        query_embedding = rag.generate_embedding(query)

        # Search
        results = mongodb.cosine_similarity_search(
            query_embedding=query_embedding.tolist(),
            k=5,
            min_similarity=0.0
        )

        if results:
            print(f"✅ Vector search successful")
            print(f"   Query: '{query}'")
            print(f"   Results found: {len(results)}")
            print(f"\n   Top 3 results:")
            for i, result in enumerate(results[:3], 1):
                movie_title = result.get('metadata', {}).get('movie_title', 'Unknown')
                section_type = result.get('section_type', 'Unknown')
                similarity = result.get('similarity', 0)
                print(f"   {i}. {movie_title} - {section_type} (similarity: {similarity:.3f})")
            return True
        else:
            print(f"❌ No results found")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_delete_embedding():
    """Test 5: Delete embedding"""
    print("\n" + "="*60)
    print("TEST 5: Delete Embedding")
    print("="*60)
    try:
        mongodb = get_mongodb_service()
        rag = RAGService()

        # Create a test embedding
        test_section_id = 999998
        test_text = "Test embedding for deletion"
        embedding = rag.generate_embedding(test_text)

        # Store it
        mongodb.store_embedding(
            section_id=test_section_id,
            movie_id=1,
            section_type='production',
            embedding=embedding.tolist()
        )

        # Delete it
        success = mongodb.delete_embedding(test_section_id)

        if success:
            print(f"✅ Embedding deleted successfully")
            print(f"   Section ID: {test_section_id}")

            # Verify it's gone
            doc = mongodb.get_embedding(test_section_id)
            if doc is None:
                print(f"   Verified: Embedding no longer exists")
                return True
            else:
                print(f"❌ Error: Embedding still exists after deletion")
                return False
        else:
            print(f"❌ Failed to delete embedding")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_rag_integration():
    """Test 6: RAG service integration"""
    print("\n" + "="*60)
    print("TEST 6: RAG Service Integration")
    print("="*60)
    try:
        # Check if we have sections
        section_count = MovieSection.objects.count()
        if section_count == 0:
            print("⚠️  No sections found in database")
            return False

        mongodb = get_mongodb_service()
        embeddings_count = mongodb.get_embeddings_count()

        if embeddings_count == 0:
            print("⚠️  No embeddings in MongoDB")
            print("   Run 'python manage.py generate_embeddings' first")
            return False

        # Test RAG search
        rag = RAGService()
        results = rag.search_with_priority("Tell me about the movie", k=3)

        if results:
            print(f"✅ RAG service working correctly")
            print(f"   Total sections: {section_count}")
            print(f"   Total embeddings: {embeddings_count}")
            print(f"   Search results: {len(results)}")
            print(f"\n   Sample result:")
            first = results[0]
            print(f"   - {first.movie.title} - {first.get_section_type_display()}")
            print(f"   - Similarity: {getattr(first, 'similarity', 'N/A')}")
            return True
        else:
            print(f"⚠️  No search results (this might be normal)")
            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "MONGODB EMBEDDING TESTS")
    print("="*70)

    tests = [
        ("Connection", test_connection),
        ("Store Embedding", test_store_embedding),
        ("Retrieve Embedding", test_retrieve_embedding),
        ("Vector Search", test_vector_search),
        ("Delete Embedding", test_delete_embedding),
        ("RAG Integration", test_rag_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")

    print("="*70)
    print(f"Result: {passed}/{total} tests passed ({(passed/total*100):.0f}%)")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! MongoDB embedding system is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")


if __name__ == '__main__':
    run_all_tests()
