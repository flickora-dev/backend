from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from chat.models import ChatConversation, ChatMessage
from services.chat_service import ChatService
import logging

logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([AllowAny])
def send_chat_message(request):
    """
    Send a chat message and get AI response
    """
    try:
        message = request.data.get('message')
        movie_id = request.data.get('movie_id')
        conversation_id = request.data.get('conversation_id')
        
        if not message:
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get or create conversation
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(id=conversation_id)
            except ChatConversation.DoesNotExist:
                conversation = None
        else:
            conversation = None
        
        if not conversation:
            conversation = ChatConversation.objects.create(
                conversation_type='movie' if movie_id else 'global',
                movie_id=movie_id
            )
        
        # Save user message
        ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=message
        )
        
        # Get AI response
        chat_service = ChatService()
        result = chat_service.chat(message, movie_id)
        
        # Save assistant message
        ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=result['message'],
            context_sections=[
                {
                    'section_id': source['section'].id,
                    'similarity': source['similarity'],
                    'movie_title': source['section'].movie.title,
                    'section_type': source['section'].get_section_type_display()
                }
                for source in result['sources']
            ]
        )
        
        # Prepare response
        response_data = {
            'message': result['message'],
            'conversation_id': conversation.id,
            'sources': [
                {
                    'section_id': source['section'].id,
                    'similarity': source['similarity'],
                    'movie_title': source['section'].movie.title,
                    'section_type': source['section'].get_section_type_display()
                }
                for source in result['sources']
            ]
        }
        
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return Response(
            {'error': 'An error occurred while processing your message'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )