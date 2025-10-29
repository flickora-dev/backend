from django.core.management.base import BaseCommand
from movies.models import Movie, Genre
from services.tmdb_service import TMDBService
import time

class Command(BaseCommand):
    help = 'Update genres for existing movies from TMDB'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--movie-id',
            type=int,
            help='Update specific movie by ID'
        )
    
    def handle(self, *args, **options):
        tmdb = TMDBService()
        
        if options['movie_id']:
            movies = Movie.objects.filter(id=options['movie_id'])
        else:
            movies = Movie.objects.all()
        
        total = movies.count()
        updated = 0
        failed = 0
        
        self.stdout.write(f"Updating genres for {total} movies...\n")
        
        for i, movie in enumerate(movies, 1):
            try:
                self.stdout.write(f"[{i}/{total}] {movie.title}...")
                
                details = tmdb.get_movie_details(movie.tmdb_id)
                
                if details and 'genres' in details:
                    movie.genres.clear()
                    
                    for genre_data in details['genres']:
                        genre, _ = Genre.objects.get_or_create(
                            tmdb_id=genre_data['id'],
                            defaults={'name': genre_data['name']}
                        )
                        movie.genres.add(genre)
                    
                    genre_count = movie.genres.count()
                    self.stdout.write(
                        self.style.SUCCESS(f"  ✓ Added {genre_count} genres")
                    )
                    updated += 1
                else:
                    self.stdout.write(
                        self.style.WARNING("  ⚠ No genre data available")
                    )
                    failed += 1
                
                time.sleep(0.3)
                
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Error: {e}")
                )
                failed += 1
        
        self.stdout.write(
            self.style.SUCCESS(f"\n✓ Updated: {updated} movies")
        )
        if failed > 0:
            self.stdout.write(
                self.style.WARNING(f"⚠ Failed: {failed} movies")
            )