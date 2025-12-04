"""
Migration script to transfer embeddings from PostgreSQL to MongoDB Atlas
Run this script BEFORE applying the database migration to remove the embedding field
"""
import os
import django
import sys

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')
django.setup()

from reports.models import MovieSection
from services.mongodb_service import get_mongodb_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_embeddings():
    """
    Migrate all embeddings from PostgreSQL pgvector to MongoDB Atlas
    """
    mongodb = get_mongodb_service()

    # Get all sections that have embeddings
    print("Fetching sections with embeddings from PostgreSQL...")
    try:
        sections = MovieSection.objects.filter(embedding__isnull=False).select_related('movie')
        total = sections.count()
    except Exception as e:
        print(f"Error: The embedding field might already be removed. Error: {e}")
        print("If you already removed the embedding field, this script cannot run.")
        print("You'll need to restore the embedding field temporarily or use a database backup.")
        return

    if total == 0:
        print("No embeddings found in PostgreSQL to migrate.")
        return

    print(f"Found {total} sections with embeddings to migrate...")

    # Prepare data for bulk insert
    embeddings_data = []
    success = 0
    failed = 0

    for i, section in enumerate(sections, 1):
        try:
            if i % 10 == 0:
                print(f"Progress: {i}/{total}")

            # Check if already exists in MongoDB
            existing = mongodb.get_embedding(section.id)
            if existing:
                print(f"  [{i}] Skipping {section.movie.title} - {section.section_type} (already in MongoDB)")
                success += 1
                continue

            # Prepare embedding data
            embedding_list = section.embedding.tolist() if hasattr(section.embedding, 'tolist') else list(section.embedding)

            embedding_doc = {
                'section_id': section.id,
                'movie_id': section.movie_id,
                'section_type': section.section_type,
                'embedding': embedding_list,
                'dimensions': len(embedding_list),
                'metadata': {
                    'movie_title': section.movie.title,
                    'section_type_display': section.get_section_type_display(),
                    'word_count': section.word_count,
                    'content_preview': section.content[:200]
                }
            }

            embeddings_data.append(embedding_doc)

            # Batch insert every 50 documents
            if len(embeddings_data) >= 50:
                count = mongodb.bulk_store_embeddings(embeddings_data)
                success += count
                failed += len(embeddings_data) - count
                embeddings_data = []
                print(f"  Batch inserted {count} embeddings")

        except Exception as e:
            logger.error(f"Error processing section {section.id}: {e}")
            failed += 1

    # Insert remaining documents
    if embeddings_data:
        count = mongodb.bulk_store_embeddings(embeddings_data)
        success += count
        failed += len(embeddings_data) - count
        print(f"  Final batch inserted {count} embeddings")

    # Summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Total sections: {total}")
    print(f"Successfully migrated: {success}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(success/total*100):.1f}%")
    print("="*60)

    # Verify MongoDB count
    mongo_count = mongodb.get_embeddings_count()
    print(f"\nTotal embeddings in MongoDB: {mongo_count}")

    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Verify embeddings in MongoDB Atlas dashboard")
    print("2. Run Django migrations: python manage.py makemigrations")
    print("3. Apply migrations: python manage.py migrate")
    print("4. Test the application thoroughly")
    print("5. Once confirmed working, you can remove pgvector from requirements.txt")


if __name__ == '__main__':
    print("="*60)
    print("EMBEDDING MIGRATION: PostgreSQL → MongoDB Atlas")
    print("="*60)
    print("\nThis script will migrate all embeddings from PostgreSQL to MongoDB.")
    print("Make sure you have:")
    print("1. MONGODB_URL configured in .env")
    print("2. MongoDB Atlas cluster accessible")
    print("3. NOT YET removed the embedding field from models.py")
    print("")

    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() == 'yes':
        migrate_embeddings()
    else:
        print("Migration cancelled.")
