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
        conversation_id = data.get('conversation_id') 
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Pobierz lub utwórz konwersację
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
        
        # Zapisz wiadomość użytkownika
        ChatMessage.objects.create(
            conversation=conversation,
            role='user',
            content=message
        )
        
        # Pobierz odpowiedź AI
        chat_service = ChatService()
        if movie_id:
            result = chat_service.chat(message, movie_id)
        else:
            # POPRAWKA: Przekaż conversation_id
            result = chat_service.global_chat.chat(message, conversation_id=conversation.id)
        
        # Zapisz odpowiedź asystenta
        ChatMessage.objects.create(
            conversation=conversation,
            role='assistant',
            content=result['message']
        )
        
        # DODAJ: Zaktualizuj referenced_movies w konwersacji
        if 'referenced_movies' in result and result['referenced_movies']:
            current_movies = conversation.referenced_movies or []
            new_movies = result['referenced_movies']
            # Zachowaj ostatnie 10 filmów
            conversation.referenced_movies = (current_movies + new_movies)[-10:]
            conversation.save()
        
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
            'sources': serialized_sources,
            'conversation_id': conversation.id  # DODAJ: Zwróć ID
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)