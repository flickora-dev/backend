from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.db.models import Q, Count
from django_filters.rest_framework import DjangoFilterBackend

from movies.models import Movie, Genre, MovieView
from movies.serializers import (
    MovieListSerializer, MovieDetailSerializer,
    GenreSerializer, MovieViewSerializer
)
from reports.models import MovieSection
from reports.serializers import MovieSectionSerializer, MovieSectionListSerializer
from chat.models import ChatConversation, ChatMessage
from chat.serializers import (
    ChatConversationSerializer, ChatRequestSerializer,
    ChatResponseSerializer
)
from services.chat_service import ChatService
from services.tmdb_service import TMDBService


class GenreViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for movie genres"""
    queryset = Genre.objects.annotate(
        movie_count=Count('movies')
    ).filter(movie_count__gt=0).order_by('name')
    serializer_class = GenreSerializer
    permission_classes = [AllowAny]


class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for movies with filtering and search"""
    queryset = Movie.objects.prefetch_related('genres', 'sections')
    permission_classes = [AllowAny]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['year', 'genres__tmdb_id']
    search_fields = ['title', 'director']
    ordering_fields = ['year', 'title', 'imdb_rating', 'created_at']
    ordering = ['-year', 'title']
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return MovieDetailSerializer
        return MovieListSerializer
    
    @action(detail=True, methods=['get'])
    def sections(self, request, pk=None):
        """Get all sections for a movie"""
        movie = self.get_object()
        sections = movie.sections.all().order_by('section_type')
        serializer = MovieSectionSerializer(sections, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def cast(self, request, pk=None):
        """Get cast for a movie from TMDB"""
        movie = self.get_object()
        tmdb = TMDBService()
        movie_data = tmdb.get_movie_details(movie.tmdb_id)
        
        if not movie_data or 'credits' not in movie_data:
            return Response({'cast': []})
        
        cast_list = []
        for person in movie_data['credits'].get('cast', [])[:10]:
            cast_list.append({
                'name': person.get('name'),
                'character': person.get('character'),
                'profile_path': f"https://image.tmdb.org/t/p/w185{person['profile_path']}" if person.get('profile_path') else None
            })
        
        return Response({'cast': cast_list})
    
    @action(detail=True, methods=['get'])
    def similar(self, request, pk=None):
        """Get similar movies from TMDB that exist in our database"""
        movie = self.get_object()
        tmdb = TMDBService()
        similar_data = tmdb.get_similar_movies(movie.tmdb_id)
        
        if not similar_data:
            return Response({'similar': []})
        
        tmdb_ids = [m['id'] for m in similar_data.get('results', [])[:20]]
        
        similar_movies = Movie.objects.filter(
            tmdb_id__in=tmdb_ids
        ).prefetch_related('genres')[:4]
        
        serializer = MovieListSerializer(similar_movies, many=True)
        return Response({'similar': serializer.data})
    
    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def view(self, request, pk=None):
        """Mark movie as viewed by current user"""
        movie = self.get_object()
        view, created = MovieView.objects.update_or_create(
            user=request.user,
            movie=movie
        )
        serializer = MovieViewSerializer(view)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def trending(self, request):
        """Get trending movies"""
        movies = self.get_queryset().order_by('-created_at')[:20]
        serializer = self.get_serializer(movies, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def recently_viewed(self, request):
        """Get recently viewed movies for current user"""
        views = MovieView.objects.filter(
            user=request.user
        ).select_related('movie').order_by('-viewed_at')[:10]
        
        movies = [view.movie for view in views]
        serializer = self.get_serializer(movies, many=True)
        return Response(serializer.data)


class MovieSectionViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for movie sections"""
    queryset = MovieSection.objects.select_related('movie')
    permission_classes = [AllowAny]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['movie', 'section_type']
    
    def get_serializer_class(self):
        if self.action == 'retrieve':
            return MovieSectionSerializer
        return MovieSectionListSerializer


class ChatViewSet(viewsets.ViewSet):
    """API endpoint for chat functionality"""
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['post'])
    def send_message(self, request):
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        message = serializer.validated_data['message']
        movie_id = serializer.validated_data.get('movie_id')
        conversation_id = serializer.validated_data.get('conversation_id')
        
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(id=conversation_id)
            except ChatConversation.DoesNotExist:
                return Response(
                    {'error': 'Conversation not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
        else:
            conversation_type = 'movie' if movie_id else 'general'
            conversation = ChatConversation.objects.create(
                conversation_type=conversation_type,
                movie_id=movie_id if movie_id else None
            )
        
        ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=message
        )
        
        chat_service = ChatService()
        result = chat_service.process_message(
            message=message,
            movie_id=movie_id,
            conversation_id=conversation.id
        )
        
        ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=result['message'],
            context_sections=[
                {
                    'section_id': source['section_id'],
                    'similarity': source['similarity'],
                    'movie_title': source['movie_title'],
                    'section_type': source['section_type']
                }
                for source in result['sources']
            ]
        )
        
        response_data = {
            'message': result['message'],
            'conversation_id': conversation.id,
            'sources': [
                {
                    'section_id': source['section_id'],
                    'similarity': source['similarity'],
                    'movie_title': source['movie_title'],
                    'section_type': source['section_type']
                }
                for source in result['sources']
            ]
        }
        
        response_serializer = ChatResponseSerializer(response_data)
        return Response(response_serializer.data)
    
    @action(detail=False, methods=['get'])
    def conversations(self, request):
        """Get user's chat conversations"""
        conversations = ChatConversation.objects.all().order_by('-updated_at')[:20]
        serializer = ChatConversationSerializer(conversations, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def conversation_detail(self, request, pk=None):
        """Get conversation with all messages"""
        try:
            conversation = ChatConversation.objects.get(id=pk)
        except ChatConversation.DoesNotExist:
            return Response(
                {'error': 'Conversation not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        serializer = ChatConversationSerializer(conversation)
        return Response(serializer.data)