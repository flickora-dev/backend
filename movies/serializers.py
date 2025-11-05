from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
from .models import Movie, Genre, MovieView

class GenreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Genre
        fields = ['id', 'tmdb_id', 'name']
        
        
class MovieListSerializer(serializers.ModelSerializer):
    genres = GenreSerializer(many=True, read_only=True)
    
    class Meta:
        model = Movie
        fields = [
            'id', 'tmdb_id', 'title', 'year', 'director',
            'genres', 'imdb_rating', 'poster_url', 'backdrop_url',
            'runtime', 'created_at'
        ]
        
class MovieDetailSerializer(serializers.ModelSerializer):
    genres = GenreSerializer(many=True, read_only=True)
    genre_list = serializers.CharField(read_only=True)
    sections_count = serializers.SerializerMethodField() 
    
    class Meta:
        model = Movie
        fields = [
            'id', 'tmdb_id', 'title', 'year', 'director',
            'genres', 'genre_list', 'imdb_rating', 'plot_summary',
            'poster_url', 'backdrop_url', 'runtime',
            'created_at', 'updated_at', 'sections_count'
        ]
        
        
    def get_sections_count(self, obj):
        return obj.sections.count()
    
    
class MovieViewSerializer(serializers.ModelSerializer):
    Movie = MovieListSerializer(read_only=True)
    
    class Meta:
        model = MovieView
        fields = ['id', 'movie', 'viewed_at']
        read_only_fields = ['viewed_at']
        

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']
        read_only_fields = ['id']
        
class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8, required=True)
    password2 = serializers.CharField(write_only=True, min_length=8, required=True)
    username = serializers.CharField(required=True, max_length=150)
    email = serializers.EmailField(required=True)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2']
        
    def validate_username(self, value):
        """Validate username uniqueness and constraints"""
        if not value:
            raise serializers.ValidationError("Username is required.")
        
        if len(value) < 3:
            raise serializers.ValidationError("Username must be at least 3 characters long.")
        
        if User.objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("A user with this username already exists.")
        
        # Check for allowed characters (alphanumeric and @/./+/-/_)
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@.+-_')
        if not all(c in allowed_chars for c in value):
            raise serializers.ValidationError("Username contains invalid characters. Only letters, numbers, and @/./+/-/_ are allowed.")
        
        return value
    
    def validate_email(self, value):
        """Validate email format and uniqueness"""
        if not value:
            raise serializers.ValidationError("Email is required.")
        
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        
        return value
    
    def validate_password(self, value):
        """Validate password using Django's password validators"""
        try:
            validate_password(value)
        except DjangoValidationError as e:
            raise serializers.ValidationError(list(e.messages))
        return value
    
    def validate(self, data):
        """Validate password match"""
        if data['password'] != data['password2']:
            raise serializers.ValidationError({
                'password2': "Passwords don't match."
            })
        return data
    
    def create(self, validated_data):
        """Create user with validated data"""
        validated_data.pop('password2')
        password = validated_data.pop('password')
        user = User.objects.create_user(password=password, **validated_data)
        return user