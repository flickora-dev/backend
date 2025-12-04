"""
Check Movie ID Type and Existence
"""
import os
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'flickora.settings')

import django
django.setup()

from flickora.mongodb import get_mongodb

def check_movie_id(target_id):
    print("=" * 80)
    print(f"Checking Movie ID: {target_id}")
    print("=" * 80)

    db = get_mongodb()
    collection = db.movie_embeddings
    
    # Try finding as int
    try:
        target_int = int(target_id)
        count_int = collection.count_documents({"movie_id": target_int})
        print(f"Count with int({target_int}): {count_int}")
    except:
        print(f"Could not convert to int")

    # Try finding as string
    target_str = str(target_id)
    count_str = collection.count_documents({"movie_id": target_str})
    print(f"Count with str('{target_str}'): {count_str}")
    
    # Get a sample to check type
    sample = collection.find_one()
    if sample:
        mid = sample.get('movie_id')
        print(f"\nSample document movie_id: {mid} (Type: {type(mid)})")
        
    # List all distinct movie_ids (limit to 10)
    print("\nFirst 10 distinct movie_ids in DB:")
    ids = collection.distinct("movie_id")
    for x in ids[:10]:
        print(f"  {x} (Type: {type(x)})")

    if target_int in ids:
        print(f"\nSUCCESS: Movie ID {target_int} found in distinct list.")
    elif target_str in ids:
         print(f"\nSUCCESS: Movie ID '{target_str}' found in distinct list.")
    else:
        print(f"\nFAILURE: Movie ID {target_id} NOT found in database.")

if __name__ == "__main__":
    check_movie_id(7)
