import requests
import json

BASE_URL = "http://localhost:8000"

def test_global_chat():
    """Test global chat (bez movie_id)"""
    print("\n" + "="*60)
    print("TEST 1: Global Chat - pytanie o filmy więzienne")
    print("="*60)
    
    response = requests.post(
        f'{BASE_URL}/chat/message/',
        json={
            'message': 'Tell me about prison movies with themes of redemption'
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status: {response.status_code}")
        print(f"\n📝 Answer:\n{data['message']}")
        print(f"\n📚 Sources ({len(data['sources'])} results):")
        for i, source in enumerate(data['sources'], 1):
            print(f"  {i}. {source.get('movie_title', 'Unknown')} - {source.get('section_type', 'Unknown')}")
            print(f"     Similarity: {source.get('similarity', 0):.3f}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def test_movie_specific_chat():
    """Test movie-specific chat (z movie_id)"""
    print("\n" + "="*60)
    print("TEST 2: Movie-Specific Chat - pytanie o konkretny film")
    print("="*60)
    
    # Pobierz listę filmów żeby znaleźć ID
    movies_response = requests.get(f'{BASE_URL}/admin/')  # lub użyj API
    
    # Dla testu użyjmy movie_id=1 (zakładając że istnieje)
    movie_id = 1
    
    response = requests.post(
        f'{BASE_URL}/chat/message/',
        json={
            'message': 'What are the main themes in this movie?',
            'movie_id': movie_id
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Status: {response.status_code}")
        print(f"\n📝 Answer:\n{data['message']}")
        print(f"\n📚 Sources ({len(data['sources'])} results):")
        for i, source in enumerate(data['sources'], 1):
            print(f"  {i}. {source.get('movie_title', 'Unknown')} - {source.get('section_type', 'Unknown')}")
            print(f"     Similarity: {source.get('similarity', 0):.3f}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def test_multiple_queries():
    """Test różnych zapytań"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Queries")
    print("="*60)
    
    queries = [
        "What movies have great cinematography?",
        "Tell me about character development in drama films",
        "Which movies explore themes of hope and freedom?",
        "Recommend movies with strong performances"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query}")
        response = requests.post(
            f'{BASE_URL}/chat/message/',
            json={'message': query}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Answer: {data['message'][:100]}...")
            print(f"   Sources: {len(data['sources'])} results")
        else:
            print(f"❌ Error: {response.status_code}")


def test_error_handling():
    """Test obsługi błędów"""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    # Test 1: Brak message
    print("\n--- Test: Missing message field")
    response = requests.post(
        f'{BASE_URL}/chat/message/',
        json={}
    )
    print(f"Status: {response.status_code} (expected: 400 or 500)")
    
    # Test 2: Nieprawidłowy movie_id
    print("\n--- Test: Invalid movie_id")
    response = requests.post(
        f'{BASE_URL}/chat/message/',
        json={
            'message': 'Test',
            'movie_id': 99999
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text[:200]}")


if __name__ == "__main__":
    print("🚀 Starting Chat API Tests")
    print("Make sure server is running: python manage.py runserver")
    
    try:
        # Sprawdź czy serwer działa
        response = requests.get(BASE_URL, timeout=2)
        print(f"\n✅ Server is running at {BASE_URL}")
    except requests.exceptions.RequestException:
        print(f"\n❌ Server not running at {BASE_URL}")
        print("Start server with: python manage.py runserver")
        exit(1)
    
    # Uruchom testy
    test_global_chat()
    test_movie_specific_chat()
    test_multiple_queries()
    test_error_handling()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)