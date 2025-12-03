# reports/management/commands/migrate_to_mongodb.py
"""
Django management command to migrate embeddings from PostgreSQL to MongoDB.

Usage:
    python manage.py migrate_to_mongodb [--batch-size 50] [--dry-run]
"""
from django.core.management.base import BaseCommand
from reports.models import MovieSection
from reports.mongodb_models import MovieSectionMongoDB
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Migrate embeddings from PostgreSQL (pgvector) to MongoDB Atlas'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of sections to process in each batch (default: 50)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be migrated without actually migrating'
        )
        parser.add_argument(
            '--movie-id',
            type=int,
            help='Migrate only sections for a specific movie ID'
        )

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        dry_run = options['dry_run']
        movie_id = options.get('movie_id')

        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS('PostgreSQL → MongoDB Migration Tool'))
        self.stdout.write(self.style.SUCCESS('=' * 80))

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No data will be written'))

        # Get sections from PostgreSQL
        queryset = MovieSection.objects.filter(embedding__isnull=False)

        if movie_id:
            queryset = queryset.filter(movie_id=movie_id)
            self.stdout.write(f'Filtering by movie_id: {movie_id}')

        total_sections = queryset.count()

        if total_sections == 0:
            self.stdout.write(self.style.WARNING('No sections with embeddings found in PostgreSQL'))
            return

        self.stdout.write(f'\nFound {total_sections} sections with embeddings in PostgreSQL')
        self.stdout.write(f'Batch size: {batch_size}\n')

        # Process in batches
        migrated = 0
        skipped = 0
        errors = 0

        for i in range(0, total_sections, batch_size):
            batch = queryset[i:i + batch_size]
            self.stdout.write(f'\nProcessing batch {i // batch_size + 1} (sections {i + 1}-{min(i + batch_size, total_sections)})...')

            for section in batch:
                try:
                    # Check if already exists in MongoDB
                    existing = MovieSectionMongoDB.get_by_movie_and_type(
                        movie_id=section.movie_id,
                        section_type=section.section_type
                    )

                    if existing:
                        self.stdout.write(
                            self.style.WARNING(
                                f'  ⚠️  Skipping {section.movie.title} - {section.section_type} (already exists in MongoDB)'
                            )
                        )
                        skipped += 1
                        continue

                    if dry_run:
                        self.stdout.write(
                            f'  [DRY RUN] Would migrate: {section.movie.title} - {section.section_type}'
                        )
                        migrated += 1
                        continue

                    # Create in MongoDB (EMBEDDING ONLY - no content!)
                    doc_id = MovieSectionMongoDB.create(
                        movie_id=section.movie_id,
                        section_type=section.section_type,
                        embedding=section.embedding
                    )

                    self.stdout.write(
                        self.style.SUCCESS(
                            f'  ✅ Migrated embedding: {section.movie.title} - {section.section_type} (ID: {doc_id})'
                        )
                    )
                    migrated += 1

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f'  ❌ Error migrating {section.movie.title} - {section.section_type}: {e}'
                        )
                    )
                    errors += 1
                    logger.error(f'Migration error for section {section.id}: {e}', exc_info=True)

        # Summary
        self.stdout.write('\n' + '=' * 80)
        self.stdout.write(self.style.SUCCESS('MIGRATION SUMMARY'))
        self.stdout.write('=' * 80)
        self.stdout.write(f'Total sections in PostgreSQL: {total_sections}')
        self.stdout.write(self.style.SUCCESS(f'✅ Migrated: {migrated}'))
        self.stdout.write(self.style.WARNING(f'⚠️  Skipped (already exist): {skipped}'))
        self.stdout.write(self.style.ERROR(f'❌ Errors: {errors}'))

        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN MODE - No data was actually written to MongoDB'))
            self.stdout.write(self.style.WARNING('Run without --dry-run to perform actual migration'))
        else:
            self.stdout.write('\n' + self.style.SUCCESS('Migration completed!'))
            self.stdout.write(self.style.WARNING('\nIMPORTANT: Create the vector search index in MongoDB Atlas:'))
            self.stdout.write('  1. Go to MongoDB Atlas → Database → Search')
            self.stdout.write('  2. Create Search Index on collection: movie_sections')
            self.stdout.write('  3. Use the following JSON definition:')
            self.stdout.write('''
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
            ''')
