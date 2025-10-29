from django.core.management.base import BaseCommand
from movies.models import Genre
import requests
from django.conf import settings

class Command(BaseCommand):
    help = 'Import movie genres from TMDB API'
    
    def handle(self, *args, **options):
        url = "https://api.themoviedb.org/3/genre/movie/list"
        params = {'api_key': settings.TMDB_API_KEY}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            imported = 0
            for genre_data in data['genres']:
                genre, created = Genre.objects.get_or_create(
                    tmdb_id=genre_data['id'],
                    defaults={'name': genre_data['name']}
                )
                
                if created:
                    imported += 1
                    self.stdout.write(f"Added: {genre.name}")
                else:
                    self.stdout.write(f"Already exists: {genre.name}")
            
            self.stdout.write(self.style.SUCCESS(f'\nSuccessfully imported {imported} new genres'))
            self.stdout.write(f'Total genres in database: {Genre.objects.count()}')
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {e}'))