from django.core.management.base import BaseCommand
from reports.models import MovieSection
from services.rag_service import RAGService
from django.db import transaction

class Command(BaseCommand):
    help = 'Regenerate embeddings for existing movie sections'
    
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
        
        # Get sections without embeddings (or all if --force)
        if options['force']:
            sections = MovieSection.objects.all()
            self.stdout.write(f"Regenerating embeddings for ALL {sections.count()} sections...")
        else:
            sections = MovieSection.objects.filter(embedding__isnull=True)
            self.stdout.write(f"Generating embeddings for {sections.count()} sections without embeddings...")
        
        if not sections.exists():
            self.stdout.write(self.style.SUCCESS('No sections need embedding generation!'))
            return
        
        batch_size = options['batch_size']
        total = sections.count()
        processed = 0
        failed = 0
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = sections[i:i + batch_size]
            
            for section in batch:
                try:
                    self.stdout.write(f"Processing: {section.movie.title} - {section.get_section_type_display()}")
                    print("1")
                    
                    # Generate embedding
                    embedding = rag.generate_embedding(section.content)
                    print("2")
                    # Save with embedding
                    section.embedding = embedding
                    print("3")
                    section.save(update_fields=['embedding'])
                    print("4")
                    
                    processed += 1
                    print("5")
                    
                    if processed % 10 == 0:
                        self.stdout.write(self.style.SUCCESS(f"  Progress: {processed}/{total}"))
                        print("6")
                    print("7")
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  ✗ Error: {e}: {e.__traceback__}"))
                    failed += 1
            
            self.stdout.write(f"Batch {i//batch_size + 1} completed")
        
        # Final stats
        self.stdout.write(self.style.SUCCESS(f"\n✓ Completed!"))
        self.stdout.write(f"  Processed: {processed}")
        self.stdout.write(f"  Failed: {failed}")
        self.stdout.write(f"  Success rate: {(processed/(processed+failed)*100):.1f}%")
        
        # Verify
        total_with_embeddings = MovieSection.objects.filter(embedding__isnull=False).count()
        total_sections = MovieSection.objects.count()
        self.stdout.write(f"\nFinal stats: {total_with_embeddings}/{total_sections} sections have embeddings")