"""
Test script to verify HNSW index is working correctly
Run: python manage.py shell < scripts/test_hnsw_index.py
"""

import time
from django.db import connection
from services.optimized_rag_service import OptimizedRAGService

# Initialize RAG service to get embedding
rag = OptimizedRAGService()

# Generate a test embedding
test_query = "What are the main themes?"
print(f"\n🔍 Generating embedding for: '{test_query}'")
test_embedding = rag.generate_embedding(test_query)

# Convert to PostgreSQL vector format
vector_str = '[' + ','.join(map(str, test_embedding)) + ']'

# Test query with EXPLAIN ANALYZE
sql = f"""
EXPLAIN ANALYZE
SELECT id, title, embedding <-> '{vector_str}'::vector AS distance
FROM reports_moviesection
WHERE embedding IS NOT NULL
ORDER BY embedding <-> '{vector_str}'::vector
LIMIT 10;
"""

print("\n📊 Running EXPLAIN ANALYZE...")
print("=" * 80)

with connection.cursor() as cursor:
    start = time.time()
    cursor.execute(sql)
    results = cursor.fetchall()
    duration = time.time() - start

    for row in results:
        print(row[0])

    print("=" * 80)
    print(f"\n⏱️  Query took: {duration:.3f}s")

# Check for index usage
explain_output = '\n'.join([row[0] for row in results])

if 'Index Scan using' in explain_output and 'hnsw' in explain_output:
    print("✅ HNSW index IS being used!")
    print("🚀 Expected performance: ~0.5-1.0s for RAG search")
elif 'Seq Scan' in explain_output:
    print("❌ HNSW index NOT being used - Sequential scan detected!")
    print("⚠️  Current performance: ~2.8-3.0s for RAG search")
    print("\n💡 To create HNSW index, run:")
    print("   CREATE INDEX idx_moviesection_embedding_hnsw")
    print("   ON reports_moviesection")
    print("   USING hnsw (embedding vector_cosine_ops)")
    print("   WITH (m = 32, ef_construction = 128);")
else:
    print("⚠️  Could not determine index usage from EXPLAIN output")
    print("\n📋 Full EXPLAIN output:")
    print(explain_output)

print("\n" + "=" * 80)

# Also test actual query performance
print("\n🧪 Testing actual query performance...")

from reports.models import MovieSection
from pgvector.django import CosineDistance

start = time.time()
results = MovieSection.objects.filter(
    embedding__isnull=False
).annotate(
    distance=CosineDistance('embedding', test_embedding)
).order_by('distance')[:10]

# Force evaluation
list(results)
duration = time.time() - start

print(f"⏱️  Actual query took: {duration:.3f}s")

if duration < 1.5:
    print("✅ Excellent! Query is fast (< 1.5s)")
elif duration < 2.5:
    print("⚠️  Query is acceptable but could be faster (1.5-2.5s)")
else:
    print("❌ Query is slow (> 2.5s) - HNSW index likely not active")
    print("   Run the CREATE INDEX command from PERFORMANCE_OPTIMIZATIONS.md")

print("\n✨ Test complete!")
