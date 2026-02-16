from django.test import TestCase
from django.contrib.auth.models import User
from django.core.cache import cache
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock

from .throttling import ChatRateThrottle, AuthRateThrottle


class AuthViewTests(APITestCase):
    """Tests for authentication views"""

    def setUp(self):
        # Clear cache to reset throttling between tests
        cache.clear()
        self.client = APIClient()
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'StrongPass123!',
            'password2': 'StrongPass123!'
        }

    def test_register_success(self):
        """Should register a new user successfully"""
        response = self.client.post('/api/auth/register/', self.user_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('tokens', response.data)
        self.assertIn('access', response.data['tokens'])
        self.assertIn('refresh', response.data['tokens'])

    def test_register_duplicate_username(self):
        """Should reject duplicate username"""
        User.objects.create_user(
            username='testuser',
            password='existing123'
        )
        response = self.client.post('/api/auth/register/', self.user_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_register_weak_password(self):
        """Should reject weak password"""
        weak_data = {
            'username': 'newuser',
            'email': 'new@example.com',
            'password': '123',
            'password2': '123'
        }
        response = self.client.post('/api/auth/register/', weak_data)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_login_success(self):
        """Should login successfully with correct credentials"""
        User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        response = self.client.post('/api/auth/login/', {
            'username': 'testuser',
            'password': 'testpass123'
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('tokens', response.data)

    def test_login_invalid_credentials(self):
        """Should reject invalid credentials"""
        User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        response = self.client.post('/api/auth/login/', {
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_login_missing_fields(self):
        """Should reject login with missing fields"""
        response = self.client.post('/api/auth/login/', {
            'username': 'testuser'
        })
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_logout_success(self):
        """Should logout successfully"""
        response = self.client.post('/api/auth/logout/', {})
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_profile_requires_auth(self):
        """Profile endpoint should require authentication"""
        response = self.client.get('/api/auth/profile/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_profile_authenticated(self):
        """Should return profile for authenticated user"""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        self.client.force_authenticate(user=user)
        response = self.client.get('/api/auth/profile/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['username'], 'testuser')

    def test_update_profile(self):
        """Should update user profile"""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
        self.client.force_authenticate(user=user)
        response = self.client.patch('/api/auth/profile/update/', {
            'email': 'newemail@example.com'
        })
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_delete_account(self):
        """Should delete user account"""
        user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        user_id = user.id
        self.client.force_authenticate(user=user)
        response = self.client.delete('/api/auth/account/delete/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(User.objects.filter(id=user_id).exists())


class ThrottlingTests(TestCase):
    """Tests for custom throttle classes"""

    def test_chat_throttle_scope(self):
        """ChatRateThrottle should have correct scope"""
        throttle = ChatRateThrottle()
        self.assertEqual(throttle.scope, 'chat')

    def test_auth_throttle_scope(self):
        """AuthRateThrottle should have correct scope"""
        throttle = AuthRateThrottle()
        self.assertEqual(throttle.scope, 'auth')


class AuthThrottlingIntegrationTests(APITestCase):
    """Integration tests for auth throttling"""

    def setUp(self):
        # Clear cache before throttling tests
        cache.clear()
        self.client = APIClient()

    def test_login_throttle_exceeded(self):
        """Should throttle after too many login attempts"""
        # Create a user
        User.objects.create_user(username='testuser', password='testpass123')

        # Make multiple failed login attempts (auth rate is 5/minute)
        for i in range(6):
            self.client.post('/api/auth/login/', {
                'username': 'testuser',
                'password': 'wrongpassword'
            })

        # Next request should be throttled
        response = self.client.post('/api/auth/login/', {
            'username': 'testuser',
            'password': 'wrongpassword'
        })
        # Should be 429 Too Many Requests
        self.assertEqual(response.status_code, status.HTTP_429_TOO_MANY_REQUESTS)


class TMDBServiceTests(TestCase):
    """Tests for TMDB service"""

    @patch('services.tmdb_service.requests.get')
    @patch('services.tmdb_service.cache')
    def test_get_movie_details_cache_miss(self, mock_cache, mock_get):
        """Should fetch from API and cache on cache miss"""
        from services.tmdb_service import TMDBService

        mock_cache.get.return_value = None  # Cache miss
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'id': 27205,
            'title': 'Inception',
            'overview': 'A thief who steals...'
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        service = TMDBService()
        result = service.get_movie_details(27205)

        self.assertEqual(result['title'], 'Inception')
        mock_cache.set.assert_called_once()

    @patch('services.tmdb_service.cache')
    def test_get_movie_details_cache_hit(self, mock_cache):
        """Should return cached data on cache hit"""
        from services.tmdb_service import TMDBService

        cached_data = {
            'id': 27205,
            'title': 'Inception (cached)',
        }
        mock_cache.get.return_value = cached_data

        service = TMDBService()
        result = service.get_movie_details(27205)

        self.assertEqual(result['title'], 'Inception (cached)')

    @patch('services.tmdb_service.requests.get')
    @patch('services.tmdb_service.cache')
    def test_get_movie_details_api_error(self, mock_cache, mock_get):
        """Should return None on API error"""
        from services.tmdb_service import TMDBService
        import requests

        mock_cache.get.return_value = None
        mock_get.side_effect = requests.RequestException("API Error")

        service = TMDBService()
        result = service.get_movie_details(27205)

        self.assertIsNone(result)

    @patch('services.tmdb_service.requests.get')
    @patch('services.tmdb_service.cache')
    def test_get_similar_movies_cached(self, mock_cache, mock_get):
        """Should cache similar movies results"""
        from services.tmdb_service import TMDBService

        mock_cache.get.return_value = None
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'results': [
                {'id': 1, 'title': 'Similar Movie 1'},
                {'id': 2, 'title': 'Similar Movie 2'},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        service = TMDBService()
        result = service.get_similar_movies(27205)

        self.assertEqual(len(result['results']), 2)
        mock_cache.set.assert_called_once()


class MovieViewSetTests(APITestCase):
    """Tests for Movie ViewSet"""

    def setUp(self):
        self.client = APIClient()

    def test_movies_list_public(self):
        """Movie list should be publicly accessible"""
        response = self.client.get('/api/movies/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_genres_list_public(self):
        """Genre list should be publicly accessible"""
        response = self.client.get('/api/genres/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)

    def test_favorite_requires_auth(self):
        """Favorite endpoint should require authentication"""
        response = self.client.post('/api/movies/1/favorite/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_recently_viewed_requires_auth(self):
        """Recently viewed should require authentication"""
        response = self.client.get('/api/movies/recently_viewed/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class HealthCheckTests(APITestCase):
    """Tests for health check endpoint"""

    def test_health_check(self):
        """Health endpoint should return OK"""
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
