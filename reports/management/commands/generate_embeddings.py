from django.core.management.base import BaseCommand
from reports.models import MovieSection
import numpy as np
import logging
from services.mongodb_service import get_mongodb_service

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate embeddings for sections and store in MongoDB'

    def add_arguments(self, parser):
        parser.add_argument('--section-id', type=int, help='Generate for specific section')
        parser.add_argument('--movie-id', type=int, help='Generate for specific movie')
        parser.add_argument('--force', action='store_true', help='Regenerate all embeddings')

    def handle(self, *args, **options):
        # Import here to avoid loading model on Django startup
        from sentence_transformers import SentenceTransformer

        # Load model once
        self.stdout.write("Loading embedding model...")
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.stdout.write(self.style.SUCCESS("[OK] Model loaded"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to load model: {e}"))
            return

        # Get MongoDB service
        mongodb = get_mongodb_service()

        # Get sections to process
        if options['section_id']:
            sections = MovieSection.objects.filter(id=options['section_id'])
        elif options['movie_id']:
            sections = MovieSection.objects.filter(movie_id=options['movie_id'])
        else:
            # Get all sections
            sections = MovieSection.objects.all()
            if not options['force']:
                # Filter out sections that already have embeddings in MongoDB
                existing_section_ids = set()
                for section in sections:
                    if mongodb.get_embedding(section.id):
                        existing_section_ids.add(section.id)
                sections = [s for s in sections if s.id not in existing_section_ids]
            else:
                sections = list(sections)

        total = len(sections)
        if total == 0:
            self.stdout.write(self.style.WARNING("No sections to process"))
            return

        self.stdout.write(f"\nProcessing {total} sections...\n")

        success = 0
        failed = 0

        for i, section in enumerate(sections, 1):
            try:
                self.stdout.write(f"[{i}/{total}] {section.movie.title} - {section.get_section_type_display()}")

                # Generate embedding
                embedding = model.encode(
                    section.content,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

                # Store in MongoDB
                success_stored = mongodb.store_embedding(
                    section_id=section.id,
                    movie_id=section.movie_id,
                    section_type=section.section_type,
                    embedding=embedding.astype('float32').tolist(),
                    metadata={
                        'movie_title': section.movie.title,
                        'section_type_display': section.get_section_type_display(),
                        'word_count': section.word_count,
                        'content_preview': section.content[:200]
                    }
                )

                if success_stored:
                    success += 1
                    self.stdout.write(self.style.SUCCESS(f"  [OK] Generated and stored in MongoDB ({len(embedding)} dims)"))
                else:
                    failed += 1
                    self.stdout.write(self.style.ERROR(f"  [FAIL] Failed to store in MongoDB"))

            except Exception as e:
                failed += 1
                self.stdout.write(self.style.ERROR(f"  [ERROR] {e}"))

        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.SUCCESS(f"[OK] Success: {success}"))
        if failed > 0:
            self.stdout.write(self.style.ERROR(f"[FAIL] Failed: {failed}"))
        self.stdout.write(f"Success rate: {(success/total*100):.1f}%")
        self.stdout.write("="*60)