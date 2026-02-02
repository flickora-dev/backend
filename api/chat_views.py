from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from chat.models import ChatConversation, ChatMessage
from chat.validators import (
    sanitize_message, validate_message, check_prompt_injection,
    prepare_user_message_for_llm
)
from services.chat_service import ChatService
import logging

logger = logging.getLogger(__name__)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def send_chat_message(request):
    """
    Send a chat message and get AI response.
    Requires authentication.
    """
    try:
        raw_message = request.data.get('message', '')
        movie_id = request.data.get('movie_id')
        conversation_id = request.data.get('conversation_id')

        # Validate message
        is_valid, error_msg = validate_message(raw_message)
        if not is_valid:
            return Response(
                {'error': error_msg},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Sanitize message
        message = sanitize_message(raw_message)

        # Check for potential prompt injection (log but don't block)
        is_suspicious, matched_pattern = check_prompt_injection(message)
        if is_suspicious:
            logger.warning(
                f"Potential prompt injection detected from user {request.user.id}: "
                f"pattern='{matched_pattern}', message='{message[:100]}...'"
            )

        # Convert movie_id to int if it's a string
        if movie_id is not None:
            try:
                movie_id = int(movie_id)
                logger.info(f"Chat request - user: {request.user.id}, movie_id: {movie_id}, message: '{message[:50]}...'")
            except (ValueError, TypeError):
                logger.warning(f"Invalid movie_id: {movie_id}")
                movie_id = None

        # Get or create conversation
        if conversation_id:
            try:
                conversation = ChatConversation.objects.get(
                    id=conversation_id,
                    user=request.user  # Ensure user owns the conversation
                )
            except ChatConversation.DoesNotExist:
                # Don't reveal if conversation exists but belongs to another user
                conversation = None
        else:
            conversation = None

        if not conversation:
            conversation = ChatConversation.objects.create(
                user=request.user,
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
        
        # Save assistant message and mark as read (since user is actively chatting)
        ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=result['message'],
            read=True,  # Mark as read since user is actively in this conversation
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
        
        # Update conversation timestamp
        conversation.updated_at = timezone.now()
        conversation.save(update_fields=['updated_at'])

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