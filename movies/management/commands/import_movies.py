from django.core.management.base import BaseCommand
from movies.models import Movie, Genre
from services.tmdb_service import TMDBService

class Command(BaseCommand):
    help = 'Import popular movies from TMDB'
    
    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=20)
        parser.add_argument('--popular', action='store_true', help='Import popular movies')
        parser.add_argument('--top-rated', action='store_true', help='Import Top rated')
        
    def handle(self, *args, **options):
        tmdb = TMDBService()
        
        if options['popular']:
            movies_data = tmdb.get_popular_movies()
        elif options['top_rated']:
            movies_data = tmdb.get_top_rated_movies()
        else:
            movies_data = tmdb.get_popular_movies()
            
        if not movies_data:
            self.stdout.write(self.style.ERROR('Failed to fetch movies'))
            return
        
        imported_count = 0
        for movie_data in movies_data['results'][:options['count']]:
            detailed_data = tmdb.get_movie_details(movie_data['id'])
            if not detailed_data:
                continue
            
            movie, created = Movie.objects.get_or_create(
                tmdb_id=movie_data['id'],
                defaults={
                    'title': detailed_data['title'],
                    'year': int(detailed_data['release_date'][:4]) if detailed_data.get('release_date') else 2024,
                    'director': self.get_director(detailed_data),
                    'plot_summary': detailed_data.get('overview', ''),
                    'runtime': detailed_data.get('runtime'),
                    'imdb_rating': detailed_data.get('vote_average'),
                    'poster_url': f"https://image.tmdb.org/t/p/w500{detailed_data['poster_path']}" if detailed_data.get('poster_path') else '',
                    'backdrop_url': f"https://image.tmdb.org/t/p/w1280{detailed_data['backdrop_path']}" if detailed_data.get('backdrop_path') else '',
                }
            )
            
            for genre_data in detailed_data.get('genres', []):
                genre, _ = Genre.objects.get_or_create(
                    tmdb_id=genre_data['id'],
                    defaults={'name': genre_data['name']}
                )
                movie.genres.add(genre)
            
            if created:
                imported_count += 1
                self.stdout.write(f"Added: {movie.title} ({movie.year})")
            else:
                self.stdout.write(f"Already exists: {movie.title}")
        
        self.stdout.write(self.style.SUCCESS(f'Successfully imported {imported_count} new movies'))
    
    def get_director(self, movie_data):
        if 'credits' in movie_data and 'crew' in movie_data['credits']:
            for person in movie_data['credits']['crew']:
                if person['job'] == 'Director':
                    return person['name']
        return 'Unknown Director'