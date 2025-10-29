from rest_framework import serializers
from .models import MovieSection


class MovieSectionSerializer(serializers.ModelSerializer):
    section_type_display = serializers.CharField(source='get_section_type_display', read_only=True)
    movie_title = serializers.CharField(source='movie.title', read_only=True)
    
    class Meta:
        model = MovieSection
        fields = [
            'id', 'movie', 'movie_title', 'section_type', 
            'section_type_display', 'content', 'word_count',
            'key_topics', 'generated_at'
        ]
        read_only_fields = ['word_count', 'generated_at']   
        
        
class MovieSectionListSerializer(serializers.ModelSerializer):
    section_type_display = serializers.CharField(source='get_section_type_display', read_only=True)

    class Meta:
        model = MovieSection
        fields = [
            'id', 'section_type', 'section_type_display',
            'word_count', 'generated_at'
        ]