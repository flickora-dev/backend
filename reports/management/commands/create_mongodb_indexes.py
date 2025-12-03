# reports/management/commands/create_mongodb_indexes.py
"""
Django management command to create MongoDB indexes.

Usage:
    python manage.py create_mongodb_indexes
"""
from django.core.management.base import BaseCommand
from reports.mongodb_models import MovieSectionMongoDB


class Command(BaseCommand):
    help = 'Create MongoDB indexes for movie_sections collection'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS('MongoDB Index Creation Tool'))
        self.stdout.write(self.style.SUCCESS('=' * 80))

        self.stdout.write('\nCreating standard indexes for movie_sections collection...')

        try:
            MovieSectionMongoDB.create_indexes()
            self.stdout.write(self.style.SUCCESS('\n✅ Standard indexes created successfully!\n'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\n❌ Error creating indexes: {e}\n'))
            return

        # Instructions for vector search index
        self.stdout.write('=' * 80)
        self.stdout.write(self.style.WARNING('IMPORTANT: Vector Search Index'))
        self.stdout.write('=' * 80)
        self.stdout.write(
            '\nThe vector search index MUST be created manually via MongoDB Atlas UI.'
        )
        self.stdout.write('Follow these steps:\n')

        self.stdout.write(self.style.SUCCESS('1. Go to MongoDB Atlas Dashboard'))
        self.stdout.write('   URL: https://cloud.mongodb.com/')

        self.stdout.write(self.style.SUCCESS('\n2. Navigate to your cluster'))
        self.stdout.write('   → Select your project')
        self.stdout.write('   → Select your cluster')

        self.stdout.write(self.style.SUCCESS('\n3. Go to Search tab'))
        self.stdout.write('   → Click "Search" in the left sidebar')
        self.stdout.write('   → Click "Create Search Index"')

        self.stdout.write(self.style.SUCCESS('\n4. Choose "JSON Editor" configuration'))

        self.stdout.write(self.style.SUCCESS('\n5. Select database and collection:'))
        self.stdout.write('   → Database: flickora (or your MONGODB_DATABASE value)')
        self.stdout.write('   → Collection: movie_embeddings')

        self.stdout.write(self.style.SUCCESS('\n6. Enter index name:'))
        self.stdout.write('   → Index Name: vector_index')

        self.stdout.write(self.style.SUCCESS('\n7. Paste this JSON configuration:'))
        self.stdout.write(self.style.WARNING('''
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 384,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
        '''))

        self.stdout.write(self.style.SUCCESS('8. Click "Create Search Index"'))

        self.stdout.write(self.style.SUCCESS('\n9. Wait for index to build'))
        self.stdout.write('   → Status will change from "Building" to "Active"')
        self.stdout.write('   → This may take a few minutes depending on data size')

        self.stdout.write('\n' + '=' * 80)
        self.stdout.write(self.style.SUCCESS('After creating the vector index, you can test with:'))
        self.stdout.write('   python manage.py test_mongodb_search\n')
