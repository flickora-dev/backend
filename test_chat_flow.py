"""
Test the actual chat flow to see where 0 results comes from
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')

import django
django.setup()

def test_movie_chat():
    print("=" * 80)
    print("Testing Movie Chat (chat_service.py)")
    print("=" * 80)

    from services.chat_service import ChatService

    chat = ChatService()
    movie_id = 3  # Test with a specific movie

    query = "What are the main themes?"
    print(f"\nQuery: '{query}'")
    print(f"Movie ID: {movie_id}")

    try:
        response = chat.get_answer(query, movie_id)
        print(f"\n[OK] Got response ({len(response)} chars)")
        print(f"\nResponse preview: {response[:200]}...")
    except Exception as e:
        print(f"\n[ERROR] Chat failed: {e}")
        import traceback
        traceback.print_exc()

def test_global_chat():
    print("\n" + "=" * 80)
    print("Testing Global Chat (global_chat_service.py)")
    print("=" * 80)

    from services.global_chat_service import GlobalChatService

    chat = GlobalChatService()

    query = "Tell me about science fiction movies"
    print(f"\nQuery: '{query}'")

    try:
        response = chat.get_answer(query, conversation_history=[])
        print(f"\n[OK] Got response ({len(response)} chars)")
        print(f"\nResponse preview: {response[:200]}...")
    except Exception as e:
        print(f"\n[ERROR] Global chat failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing both chat services...\n")
    test_movie_chat()
    test_global_chat()
    print("\n" + "=" * 80)
    print("Tests Complete")
    print("=" * 80)
