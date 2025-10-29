from rest_framework import serializers
from django.contrib.auth.models import User
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
    password = serializers.CharField(write_only=True, min_length=8)
    password2 = serializers.CharField(write_only=True, min_length=8)
    
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2']
        
    def validate(self, data):
        if data['password'] != data['password2']:
            raise serializers.ValidationError("Passwords don't match")
        return data
    
    def create(self, validated_data):
        validated_data.pop('password2')
        user = User.objects.create_user(**validated_data)
        return user