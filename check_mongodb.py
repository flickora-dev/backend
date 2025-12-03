"""
Quick script to check MongoDB embeddings status
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

def check_mongodb_status():
    """Check MongoDB embeddings status"""
    print("=" * 80)
    print("MongoDB Embeddings Status Check")
    print("=" * 80)

    # Connect to MongoDB
    from flickora.mongodb import get_mongodb
    db = get_mongodb()
    collection = db.movie_embeddings

    # Count embeddings
    total_embeddings = collection.count_documents({})
    print(f"\n✓ Total embeddings in MongoDB: {total_embeddings}")

    # Count sections in PostgreSQL
    total_sections = MovieSection.objects.count()
    print(f"✓ Total sections in PostgreSQL: {total_sections}")

    if total_embeddings == 0:
        print("\n" + "=" * 80)
        print("⚠️  WARNING: No embeddings found in MongoDB!")
        print("=" * 80)
        print("\nYou need to generate embeddings. Run one of these commands:")
        print("  1. Generate for all sections:")
        print("     python manage.py generate_embeddings_mongodb")
        print("\n  2. Migrate existing embeddings from PostgreSQL:")
        print("     python manage.py migrate_to_mongodb")
        print("\n  3. Use Django admin to regenerate embeddings for specific sections")
        print("=" * 80)
    else:
        print(f"\n✓ MongoDB has {total_embeddings} embeddings")

        # Check if vector index exists
        indexes = list(collection.list_indexes())
        has_vector_index = any('vector' in idx.get('name', '').lower() for idx in indexes)

        if has_vector_index:
            print("✓ Vector search index exists")
        else:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: Vector search index NOT found!")
            print("=" * 80)
            print("\nYou need to create a vector search index in MongoDB Atlas.")
            print("Run: python manage.py create_mongodb_indexes")
            print("Then follow the instructions to create the vector index in Atlas UI.")
            print("=" * 80)

    print("\n")

if __name__ == "__main__":
    check_mongodb_status()
