# reports/management/commands/migrate_section_structure.py

from django.core.management.base import BaseCommand
from django.db import transaction
from reports.models import MovieSection
from movies.models import Movie
from services.openrouter_service import OpenRouterService
from services.rag_service import RAGService
import time


class Command(BaseCommand):
    help = 'Migrate existing 6-section structure to new 8-10 section structure'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--rename-only',
            action='store_true',
            help='Only rename existing sections without generating new ones'
        )
        parser.add_argument(
            '--generate-missing',
            action='store_true',
            help='Generate missing sections for all movies'
        )
        parser.add_argument(
            '--movie-id',
            type=int,
            help='Process specific movie only'
        )
    
    def handle(self, *args, **options):
        self.stdout.write("="*70)
        self.stdout.write("SECTION STRUCTURE MIGRATION")
        self.stdout.write("="*70)
        
        # Step 1: Rename existing sections
        self.stdout.write("\n📝 Step 1: Renaming existing sections...")
        renamed_count = self._rename_existing_sections()
        self.stdout.write(self.style.SUCCESS(f"✓ Renamed {renamed_count} sections"))
        
        if options['rename_only']:
            self.stdout.write("\n✓ Rename complete. Use --generate-missing to add new sections.")
            return
        
        # Step 2: Generate missing sections
        if options['generate_missing']:
            self.stdout.write("\n📝 Step 2: Generating missing sections...")
            
            if options['movie_id']:
                movies = Movie.objects.filter(id=options['movie_id'])
            else:
                movies = Movie.objects.all()
            
            generated_count = self._generate_missing_sections(movies)
            self.stdout.write(self.style.SUCCESS(f"✓ Generated {generated_count} new sections"))
        
        # Step 3: Summary
        self._print_summary()
    
    def _rename_existing_sections(self):
        """Rename old section types to new ones"""
        mapping = {
            'basic_info': 'production',
            'cast_performances': 'cast_crew',
            'character_analysis': 'characters',
            'thematic_artistic': 'themes',
            'critical_reception': 'reception',
            'legacy_impact': 'legacy',
        }
        
        total_renamed = 0
        for old_type, new_type in mapping.items():
            count = MovieSection.objects.filter(section_type=old_type).update(section_type=new_type)
            if count > 0:
                self.stdout.write(f"  {old_type} → {new_type}: {count} sections")
                total_renamed += count
        
        return total_renamed
    
    def _generate_missing_sections(self, movies):
        """Generate new required sections for movies"""
        openrouter = OpenRouterService()
        rag = RAGService()
        
        # Priority order for generation
        new_sections = [
            'plot_structure',      # CRITICAL - most important
            'visual_technical',    # Split from old thematic_artistic
        ]
        
        total_generated = 0
        total_movies = movies.count()
        
        for i, movie in enumerate(movies, 1):
            self.stdout.write(f"\n[{i}/{total_movies}] Processing: {movie.title}")
            
            movie_data = {
                'title': movie.title,
                'year': movie.year,
                'director': movie.director,
                'genres': ', '.join([g.name for g in movie.genres.all()]),
                'plot_summary': movie.plot_summary
            }
            
            for section_type in new_sections:
                if MovieSection.objects.filter(movie=movie, section_type=section_type).exists():
                    self.stdout.write(f"  ✓ {section_type}: already exists")
                    continue
                
                try:
                    self.stdout.write(f"  🔄 Generating {section_type}...")
                    
                    # Generate content
                    content = openrouter.generate_movie_section(movie_data, section_type)
                    
                    if content:
                        # Generate embedding
                        embedding = rag.generate_embedding(content)
                        
                        # Save
                        MovieSection.objects.create(
                            movie=movie,
                            section_type=section_type,
                            content=content,
                            embedding=embedding
                        )
                        
                        total_generated += 1
                        word_count = len(content.split())
                        self.stdout.write(self.style.SUCCESS(
                            f"  ✓ {section_type}: {word_count} words"
                        ))
                    else:
                        self.stdout.write(self.style.ERROR(f"  ✗ Failed to generate {section_type}"))
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  ✗ Error: {e}"))
        
        return total_generated
    
    def _print_summary(self):
        """Print migration summary"""
        self.stdout.write("\n" + "="*70)
        self.stdout.write("MIGRATION SUMMARY")
        self.stdout.write("="*70)
        
        total_movies = Movie.objects.count()
        total_sections = MovieSection.objects.count()
        
        self.stdout.write(f"\nTotal movies: {total_movies}")
        self.stdout.write(f"Total sections: {total_sections}")
        self.stdout.write(f"Average sections per movie: {total_sections/total_movies:.1f}")
        
        # Section type distribution
        self.stdout.write("\nSection distribution:")
        from django.db.models import Count
        section_counts = MovieSection.objects.values('section_type').annotate(
            count=Count('id')
        ).order_by('section_type')
        
        for item in section_counts:
            section_name = dict(MovieSection.SECTION_TYPES).get(item['section_type'], item['section_type'])
            self.stdout.write(f"  {section_name}: {item['count']}")
        
        # Movies with complete sections (8 core sections)
        complete_movies = Movie.objects.annotate(
            section_count=Count('sections')
        ).filter(section_count__gte=8).count()
        
        self.stdout.write(f"\nMovies with 8+ sections: {complete_movies}/{total_movies}")
        self.stdout.write("="*70)