from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from .models import ChatConversation, ChatMessage
from services.chat_service import ChatService
from django.views.decorators.csrf import csrf_exempt 

@csrf_exempt
@require_POST
def chat_message(request):
    try:
        data = json.loads(request.body)
        message = data.get('message')
        movie_id = data.get('movie_id')
        
        # Validation
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Create conversation
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
            content=result['message']
        )
        
        # Serialize sources (convert MovieSection objects to dicts)
        serialized_sources = [
            {
                'section_id': source['section_id'],
                'similarity': source['similarity'],
                'movie_title': source['movie_title'],
                'section_type': source['section_type']
            }
            for source in result['sources']
        ]
        
        return JsonResponse({
            'message': result['message'],
            'sources': serialized_sources
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)