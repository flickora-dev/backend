from django.urls import path
from . import views

urlpatterns = [
    path('send/', views.chat_message, name='chat_message'),
    path('send/stream/', views.chat_message_stream, name='chat_message_stream'),
    path('conversations/', views.get_conversations, name='get_conversations'),
    path('<int:conversation_id>/conversation_detail/', views.get_conversation_detail, name='get_conversation_detail'),
    path('<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
]
