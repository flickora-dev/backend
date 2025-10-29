import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_api():
    print("🧪 Testing flickora API for N8N automation\n")
    
    # Test 1: Get movies without reports
    print("1️⃣ Fetching movies without complete reports...")
    response = requests.get(f"{BASE_URL}/api/movies-without-reports/?limit=3")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['count']} movies")
        for movie in data['movies']:
            print(f"   - {movie['title']} ({movie['year']}) - {movie['sections_count']}/6 sections")
        
        if data['count'] > 0:
            test_movie = data['movies'][0]
            print(f"\n2️⃣ Testing with movie: {test_movie['title']}")
            
            # Test 2: Check movie status
            print("\n   Checking detailed status...")
            status_response = requests.get(f"{BASE_URL}/api/movie-status/{test_movie['id']}/")
            
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   ✅ Movie has {status['total_sections']}/6 sections")
                
                # Find missing section
                missing_section = None
                for section_type, info in status['sections'].items():
                    if not info['exists']:
                        missing_section = section_type
                        break
                
                if missing_section:
                    print(f"\n3️⃣ Generating missing section: {missing_section}")
                    
                    # Test 3: Generate section
                    gen_response = requests.post(
                        f"{BASE_URL}/api/generate-section/",
                        json={
                            "movie_id": test_movie['id'],
                            "section_type": missing_section
                        }
                    )
                    
                    if gen_response.status_code == 200:
                        section_data = gen_response.json()
                        print(f"   ✅ Section generated!")
                        print(f"   Word count: {section_data['section']['word_count']}")
                        
                        # Test 4: Generate embedding
                        print(f"\n4️⃣ Generating embedding...")
                        time.sleep(1)
                        
                        emb_response = requests.post(
                            f"{BASE_URL}/api/generate-embedding/",
                            json={
                                "section_id": section_data['section']['id']
                            }
                        )
                        
                        if emb_response.status_code == 200:
                            emb_data = emb_response.json()
                            print(f"   ✅ Embedding generated!")
                            print(f"   Dimensions: {emb_data['embedding_dimensions']}")
                            
                            print("\n🎉 All tests passed! API is ready for N8N automation.")
                        else:
                            print(f"   ❌ Embedding failed: {emb_response.text}")
                    else:
                        print(f"   ❌ Generation failed: {gen_response.text}")
                else:
                    print("   ℹ️  All sections already exist for this movie")
            else:
                print(f"   ❌ Status check failed: {status_response.text}")
    else:
        print(f"❌ API request failed: {response.text}")

if __name__ == "__main__":
    print("Make sure Django server is running: python manage.py runserver\n")
    time.sleep(2)
    test_api()