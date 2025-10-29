from django.core.management.base import BaseCommand
from movies.models import Movie
from reports.models import MovieSection
from services.openrouter_service import OpenRouterService
from services.rag_service import RAGService
import time

class Command(BaseCommand):
    help = 'Generate AI reports for movies with automatic embedding generation'
    
    def add_arguments(self, parser):
        parser.add_argument('--movie-id', type=int, help='Generate report for specific movie')
        parser.add_argument('--all', action='store_true', help='Generate reports for all movies without reports')
        parser.add_argument('--limit', type=int, default=5, help='Limit number of movies to process')
        parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation')
    
    def handle(self, *args, **options):
        openrouter = OpenRouterService()
        rag = RAGService()
        
        if options['movie_id']:
            movies = Movie.objects.filter(id=options['movie_id'])
        elif options['all']:
            movies = Movie.objects.filter(sections__isnull=True).distinct()[:options['limit']]
        else:
            movies = Movie.objects.filter(sections__isnull=True).distinct()[:options['limit']]
        
        if not movies:
            self.stdout.write(self.style.WARNING('No movies to process'))
            return
        
        section_types = [
            'production',
            'plot_structure',
            'cast_crew',
            'characters',
            'visual_technical',
            'themes',
            'reception',
            'legacy'
        ]
        
        total_generated = 0
        
        for movie in movies:
            self.stdout.write(f"\nProcessing: {movie.title} ({movie.year})")
            
            movie_data = {
                'title': movie.title,
                'year': movie.year,
                'director': movie.director,
                'genres': ', '.join([g.name for g in movie.genres.all()]),
                'plot_summary': movie.plot_summary
            }
            
            for section_type in section_types:
                if MovieSection.objects.filter(movie=movie, section_type=section_type).exists():
                    self.stdout.write(f"  - {section_type}: already exists")
                    continue
                
                try:
                    self.stdout.write(f"  - Generating {section_type}...")
                    
                    content = openrouter.generate_movie_section(movie_data, section_type)
                    
                    if content:
                        embedding = None
                        
                        if not options['skip_embeddings']:
                            self.stdout.write(f"    - Generating embedding...")
                            embedding = rag.generate_embedding(content)
                        
                        MovieSection.objects.create(
                            movie=movie,
                            section_type=section_type,
                            content=content,
                            embedding=embedding
                        )
                        
                        total_generated += 1
                        emb_status = 'yes' if embedding is not None else 'no'
                        self.stdout.write(self.style.SUCCESS(
                            f"    ✓ Generated ({len(content.split())} words, embedding: {emb_status})"
                        ))
                    else:
                        self.stdout.write(self.style.ERROR(f"    ✗ Failed to generate"))
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"    ✗ Error: {e}"))
        
        self.stdout.write(self.style.SUCCESS(f"\nTotal sections generated: {total_generated}"))