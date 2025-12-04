from django.core.management.base import BaseCommand
from reports.models import MovieSection
from services.rag_service import RAGService
from services.mongodb_service import get_mongodb_service
from django.db import transaction

class Command(BaseCommand):
    help = 'Regenerate embeddings for existing movie sections and store in MongoDB'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Regenerate embeddings even if they already exist'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of sections to process in each batch'
        )

    def handle(self, *args, **options):
        rag = RAGService()
        mongodb = get_mongodb_service()

        # Get all sections
        sections = MovieSection.objects.all()

        if not options['force']:
            # Filter out sections that already have embeddings in MongoDB
            sections_to_process = []
            for section in sections:
                if not mongodb.get_embedding(section.id):
                    sections_to_process.append(section)
            sections = sections_to_process
            self.stdout.write(f"Generating embeddings for {len(sections)} sections without embeddings in MongoDB...")
        else:
            sections = list(sections)
            self.stdout.write(f"Regenerating embeddings for ALL {len(sections)} sections...")

        if len(sections) == 0:
            self.stdout.write(self.style.SUCCESS('No sections need embedding generation!'))
            return

        batch_size = options['batch_size']
        total = len(sections)
        processed = 0
        failed = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = sections[i:i + batch_size]

            for section in batch:
                try:
                    self.stdout.write(f"Processing: {section.movie.title} - {section.get_section_type_display()}")

                    # Generate embedding
                    embedding = rag.generate_embedding(section.content)

                    # Store in MongoDB
                    success = mongodb.store_embedding(
                        section_id=section.id,
                        movie_id=section.movie_id,
                        section_type=section.section_type,
                        embedding=embedding.tolist(),
                        metadata={
                            'movie_title': section.movie.title,
                            'section_type_display': section.get_section_type_display(),
                            'word_count': section.word_count,
                            'content_preview': section.content[:200]
                        }
                    )

                    if success:
                        processed += 1
                    else:
                        failed += 1
                        self.stdout.write(self.style.ERROR(f"  ✗ Failed to store in MongoDB"))

                    if processed % 10 == 0:
                        self.stdout.write(self.style.SUCCESS(f"  Progress: {processed}/{total}"))

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  ✗ Error: {e}"))
                    failed += 1

            self.stdout.write(f"Batch {i//batch_size + 1} completed")

        # Final stats
        self.stdout.write(self.style.SUCCESS(f"\n✓ Completed!"))
        self.stdout.write(f"  Processed: {processed}")
        self.stdout.write(f"  Failed: {failed}")
        if processed + failed > 0:
            self.stdout.write(f"  Success rate: {(processed/(processed+failed)*100):.1f}%")

        # Verify
        total_sections = MovieSection.objects.count()
        total_with_embeddings = mongodb.get_embeddings_count()
        self.stdout.write(f"\nFinal stats: {total_with_embeddings}/{total_sections} sections have embeddings in MongoDB")