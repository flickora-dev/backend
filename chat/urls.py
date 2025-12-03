from django.urls import path
from . import views

urlpatterns = [
    path('message/', views.chat_message, name='chat_message'),
    path('message/stream/', views.chat_message_stream, name='chat_message_stream'),
]
