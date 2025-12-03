# reports/management/commands/generate_embeddings_mongodb.py
"""
Django management command to generate embeddings for MongoDB movie sections.

Usage:
    python manage.py generate_embeddings_mongodb [--force] [--movie-id ID]
"""
from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from reports.mongodb_models import MovieSectionMongoDB
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate embeddings for MongoDB movie sections'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Regenerate embeddings even if they already exist'
        )
        parser.add_argument(
            '--movie-id',
            type=int,
            help='Generate embeddings only for sections of a specific movie'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of sections to process in each batch (default: 50)'
        )

    def handle(self, *args, **options):
        force = options['force']
        movie_id = options.get('movie_id')
        batch_size = options['batch_size']

        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS('MongoDB Embedding Generation Tool'))
        self.stdout.write(self.style.SUCCESS('=' * 80))

        # Load the model
        self.stdout.write('Loading sentence-transformers model...')
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.stdout.write(self.style.SUCCESS('✅ Model loaded successfully\n'))

        # Get sections that need embeddings
        collection = MovieSectionMongoDB.get_collection()

        if force:
            # Get all sections
            if movie_id:
                query = {"movie_id": movie_id}
            else:
                query = {}
            self.stdout.write(self.style.WARNING('Force mode: Regenerating ALL embeddings'))
        else:
            # Get only sections without embeddings
            if movie_id:
                query = {"movie_id": movie_id, "embedding": None}
            else:
                query = {"embedding": None}

        sections = list(collection.find(query))

        if not sections:
            self.stdout.write(self.style.WARNING('No sections found that need embeddings'))
            return

        total_sections = len(sections)
        self.stdout.write(f'Found {total_sections} sections to process')
        self.stdout.write(f'Batch size: {batch_size}\n')

        # Process in batches
        processed = 0
        errors = 0

        for i in range(0, total_sections, batch_size):
            batch = sections[i:i + batch_size]
            self.stdout.write(
                f'\nProcessing batch {i // batch_size + 1} '
                f'(sections {i + 1}-{min(i + batch_size, total_sections)})...'
            )

            for section in batch:
                try:
                    # Generate embedding
                    content = section.get('content', '')
                    if not content:
                        self.stdout.write(
                            self.style.WARNING(
                                f'  ⚠️  Skipping section with no content (ID: {section["_id"]})'
                            )
                        )
                        continue

                    embedding = model.encode(
                        content,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=False
                    )

                    # Convert to float32 for storage efficiency
                    embedding = embedding.astype('float32')

                    # Update in MongoDB
                    MovieSectionMongoDB.update_embedding(
                        movie_id=section['movie_id'],
                        section_type=section['section_type'],
                        embedding=embedding
                    )

                    # Get movie title for display (from PostgreSQL)
                    from movies.models import Movie
                    try:
                        movie = Movie.objects.get(id=section['movie_id'])
                        movie_title = movie.title
                    except Movie.DoesNotExist:
                        movie_title = f"Movie ID {section['movie_id']}"

                    self.stdout.write(
                        self.style.SUCCESS(
                            f'  ✅ Generated embedding for: {movie_title} - {section["section_type"]}'
                        )
                    )
                    processed += 1

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f'  ❌ Error processing section {section["_id"]}: {e}'
                        )
                    )
                    errors += 1
                    logger.error(f'Embedding generation error for section {section["_id"]}: {e}', exc_info=True)

        # Summary
        self.stdout.write('\n' + '=' * 80)
        self.stdout.write(self.style.SUCCESS('EMBEDDING GENERATION SUMMARY'))
        self.stdout.write('=' * 80)
        self.stdout.write(f'Total sections: {total_sections}')
        self.stdout.write(self.style.SUCCESS(f'✅ Successfully processed: {processed}'))
        self.stdout.write(self.style.ERROR(f'❌ Errors: {errors}'))
        self.stdout.write('\n' + self.style.SUCCESS('Embedding generation completed!'))
