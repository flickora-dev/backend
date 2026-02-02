import requests
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class TMDBService:
    def __init__(self):
        self.api_key = settings.TMDB_API_KEY
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p/w500"
    
    def get_movie_details(self, tmdb_id):
        """Get detailed movie information from TMDB"""
        try:
            url = f"{self.base_url}/movie/{tmdb_id}"
            params = {
                'api_key': self.api_key,
                'append_to_response': 'credits,keywords'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching movie {tmdb_id}: {e}")
            return None
    
    def get_similar_movies(self, tmdb_id):
        """Get similar movies from TMDB"""
        try:
            url = f"{self.base_url}/movie/{tmdb_id}/similar"
            params = {
                'api_key': self.api_key,
                'page': 1
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching similar movies for {tmdb_id}: {e}")
            return None
    
    def search_movies(self, query, page=1):
        """Search for movies by title"""
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': query,
                'page': page
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error searching movies '{query}': {e}")
            return None
    
    def get_popular_movies(self, page=1):
        """Get popular movies"""
        try:
            url = f"{self.base_url}/movie/popular"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching popular movies: {e}")
            return None
    
    def get_top_rated_movies(self, page=1):
        """Get top rated movies"""
        try:
            url = f"{self.base_url}/movie/top_rated"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching top rated movies: {e}")
            return None

    def get_trending_movies(self, time_window='week', page=1):
        """Get trending movies (day or week)"""
        try:
            url = f"{self.base_url}/trending/movie/{time_window}"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching trending movies: {e}")
            return None

    def get_upcoming_movies(self, page=1):
        """Get upcoming movies"""
        try:
            url = f"{self.base_url}/movie/upcoming"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching upcoming movies: {e}")
            return None

    def get_now_playing_movies(self, page=1):
        """Get now playing movies in theaters"""
        try:
            url = f"{self.base_url}/movie/now_playing"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching now playing movies: {e}")
            return None

    def get_new_releases(self, days=7, page=1):
        """Get movies released in the last N days"""
        from datetime import datetime, timedelta

        today = datetime.now().date()
        start_date = today - timedelta(days=days)

        try:
            url = f"{self.base_url}/discover/movie"
            params = {
                'api_key': self.api_key,
                'page': page,
                'primary_release_date.gte': start_date.isoformat(),
                'primary_release_date.lte': today.isoformat(),
                'sort_by': 'primary_release_date.desc',
                'vote_count.gte': 10  # Filter out obscure movies
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching new releases: {e}")
            return None