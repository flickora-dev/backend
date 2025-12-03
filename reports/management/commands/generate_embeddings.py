from django.core.management.base import BaseCommand
from reports.models import MovieSection
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate embeddings for sections without embeddings'
    
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
            self.stdout.write(self.style.SUCCESS("✓ Model loaded"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to load model: {e}"))
            return
        
        # Get sections to process
        if options['section_id']:
            sections = MovieSection.objects.filter(id=options['section_id'])
        elif options['movie_id']:
            sections = MovieSection.objects.filter(movie_id=options['movie_id'])
        elif options['force']:
            sections = MovieSection.objects.all()
        else:
            sections = MovieSection.objects.filter(embedding__isnull=True)
        
        total = sections.count()
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
                
                # Save
                section.embedding = embedding.astype('float32')
                section.save(update_fields=['embedding'])
                
                success += 1
                self.stdout.write(self.style.SUCCESS(f"  ✓ Generated ({len(embedding)} dims)"))
                
            except Exception as e:
                failed += 1
                self.stdout.write(self.style.ERROR(f"  ✗ Error: {e}"))
        
        # Summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.SUCCESS(f"✓ Success: {success}"))
        if failed > 0:
            self.stdout.write(self.style.ERROR(f"✗ Failed: {failed}"))
        self.stdout.write(f"Success rate: {(success/total*100):.1f}%")
        self.stdout.write("="*60)