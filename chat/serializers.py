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
    movie_title = serializers.CharField(source='movie.title', read_only=True, allow_null=True)
    
    class Meta:
        model = ChatConversation
        fields = [
            'id', 'conversation_type', 'movie', 'movie_title',
            'created_at', 'updated_at', 'messages'
        ]
        read_only_fields = ['created_at', 'updated_at']


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