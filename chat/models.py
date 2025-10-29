from django.db import models
from movies.models import Movie


class ChatConversation(models.Model):
    CONVERSATION_TYPES = [
        ('global', 'Global Chat'),
        ('movie', 'Movie-Specific Chat'),
    ]
    
    conversation_type = models.CharField(max_length=20, choices=CONVERSATION_TYPES)
    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='conversations'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        
    def __str__(self):
        if self.conversation_type == 'movie' and self.movie:
            return f"Chat about {self.movie.title}"
        return f"Global Chat #{self.id}"
    
    
class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]
    
    conversation = models.ForeignKey(
        ChatConversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    context_sections = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
        
    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."