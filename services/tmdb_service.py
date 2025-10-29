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