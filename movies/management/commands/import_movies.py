from django.core.management.base import BaseCommand
from movies.models import Movie, Genre
from services.tmdb_service import TMDBService
from django.conf import settings
import requests

class Command(BaseCommand):
    help = 'Import popular movies from TMDB'
    
    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=20)
        parser.add_argument('--popular', action='store_true', help='Import popular movies')
        parser.add_argument('--top-rated', action='store_true', help='Import top rated movies')
        parser.add_argument('--trending', action='store_true', help='Import trending movies this week')
        parser.add_argument('--upcoming', action='store_true', help='Import upcoming movies')
        parser.add_argument('--now-playing', action='store_true', help='Import now playing in theaters')
        parser.add_argument('--new-releases', action='store_true', help='Import movies released in the last 7 days')
        parser.add_argument('--days', type=int, default=7, help='Number of days to look back for new releases')
        
    def handle(self, *args, **options):
        tmdb = TMDBService()
        target_count = options['count']
        imported_count = 0
        skipped_count = 0
        page = 1

        self.stdout.write(f"Importing up to {target_count} NEW movies...")

        while imported_count < target_count:
            # Fetch movies for current page
            if options['new_releases']:
                movies_data = tmdb.get_new_releases(days=options['days'], page=page)
            elif options['trending']:
                movies_data = tmdb.get_trending_movies(page=page)
            elif options['upcoming']:
                movies_data = tmdb.get_upcoming_movies(page=page)
            elif options['now_playing']:
                movies_data = tmdb.get_now_playing_movies(page=page)
            elif options['top_rated']:
                movies_data = tmdb.get_top_rated_movies(page=page)
            elif options['popular']:
                movies_data = tmdb.get_popular_movies(page=page)
            else:
                movies_data = tmdb.get_popular_movies(page=page)

            if not movies_data or not movies_data.get('results'):
                self.stdout.write(self.style.WARNING('No more movies available'))
                break

            for movie_data in movies_data['results']:
                if imported_count >= target_count:
                    break

                tmdb_id = movie_data['id']

                # Check if movie already exists - skip fetching details if it does
                if Movie.objects.filter(tmdb_id=tmdb_id).exists():
                    skipped_count += 1
                    self.stdout.write(self.style.WARNING(f"Skipped (exists): {movie_data.get('title')}"))
                    continue

                # Fetch detailed data only for new movies
                detailed_data = tmdb.get_movie_details(tmdb_id)
                if not detailed_data:
                    self.stdout.write(self.style.ERROR(f"Failed to fetch details for TMDB ID {tmdb_id}"))
                    continue

                movie, created = Movie.objects.get_or_create(
                    tmdb_id=tmdb_id,
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
                    self.stdout.write(self.style.SUCCESS(f"✓ Added: {movie.title} ({movie.year})"))

            page += 1

        # Trigger n8n workflow after import if any movies were added
        if imported_count > 0:
            self.trigger_n8n_workflow()

        self.stdout.write(self.style.SUCCESS(f'\nImport complete: {imported_count} new movies imported, {skipped_count} skipped'))

    def trigger_n8n_workflow(self):
        """Trigger n8n workflow to process movies"""
        webhook_url = getattr(settings, 'N8N_WEBHOOK_URL', None)

        if not webhook_url:
            self.stdout.write(self.style.WARNING('N8N_WEBHOOK_URL not configured - skipping workflow trigger'))
            return

        try:
            response = requests.post(webhook_url, json={'trigger': 'import_complete'}, timeout=5)
            if response.status_code in [200, 201]:
                self.stdout.write(self.style.SUCCESS('✓ n8n workflow triggered'))
            else:
                self.stdout.write(self.style.WARNING(f'n8n workflow trigger failed: {response.status_code}'))
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'Failed to trigger n8n workflow: {e}'))
    
    def get_director(self, movie_data):
        if 'credits' in movie_data and 'crew' in movie_data['credits']:
            for person in movie_data['credits']['crew']:
                if person['job'] == 'Director':
                    return person['name']
        return 'Unknown Director'