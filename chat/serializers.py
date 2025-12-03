from rest_framework import serializers
from .models import ChatConversation, ChatMessage


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'role', 'content', 'context_sections', 'created_at'
        ]
        read_only_fields = ['created_at']


class ChatConversationSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    movie = serializers.SerializerMethodField()

    class Meta:
        model = ChatConversation
        fields = [
            'id', 'conversation_type', 'movie',
            'created_at', 'updated_at', 'messages', 'referenced_movies'
        ]
        read_only_fields = ['created_at', 'updated_at']

    def get_movie(self, obj):
        """Include movie details if this is a movie conversation"""
        if obj.movie:
            return {
                'id': obj.movie.id,
                'title': obj.movie.title,
                'year': obj.movie.year,
                'poster_url': obj.movie.poster_url,
            }
        return None


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for chat requests"""
    message = serializers.CharField(required=True, max_length=2000)
    movie_id = serializers.IntegerField(required=False, allow_null=True)
    conversation_id = serializers.IntegerField(required=False, allow_null=True)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat responses"""
    message = serializers.CharField()
    conversation_id = serializers.IntegerField()
    sources = serializers.ListField(
        child=serializers.DictField(),
        required=False
    )