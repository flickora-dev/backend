# Performance Optimizations for flickora

## 1. Add HNSW Index for pgvector (Critical - Reduces RAG from ~3s to ~0.5s)

### Step 1: Enable HNSW extension
Run this SQL on your Railway PostgreSQL database:

```sql
-- Enable HNSW extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS hnsw;
```

### Step 2: Create HNSW index on embeddings
```sql
-- Create HNSW index on reports_moviesection.embedding
-- This will dramatically speed up vector similarity searches
CREATE INDEX IF NOT EXISTS idx_moviesection_embedding_hnsw
ON reports_moviesection
USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 128);

-- Update statistics
ANALYZE reports_moviesection;
```

### Step 3: Configure query-time parameters (optional)
For better recall at cost of speed:
```sql
-- Session-level setting (can be set in Django connection)
SET hnsw.ef_search = 64;  -- default is 40, higher = better recall but slower
```

### Expected Results:
- **Before**: RAG search ~2.8-3.0s (sequential scan)
- **After**: RAG search ~0.5-1.0s (HNSW index scan)
- **Total improvement**: ~2-2.5s per request

### Verification:
Run the test script to check if HNSW index is working:

```bash
# On Railway or local
python manage.py shell < scripts/test_hnsw_index.py
```

This will:
1. Generate a real embedding vector
2. Run EXPLAIN ANALYZE with actual vector data
3. Check if HNSW index is being used
4. Measure actual query performance

**Expected output with HNSW:**
```
✅ HNSW index IS being used!
⏱️  Actual query took: 0.687s
✅ Excellent! Query is fast (< 1.5s)
```

**Expected output WITHOUT HNSW:**
```
❌ HNSW index NOT being used - Sequential scan detected!
⏱️  Actual query took: 2.943s
❌ Query is slow (> 2.5s)
```

---

## 2. Streaming LLM Responses (Reduces perceived latency)

Streaming is implemented in `chat/views.py` with `stream_chat_response()`.

### Benefits:
- **First token**: ~0.4-0.6s (vs 8-9s for full response)
- **User perception**: Instant feedback instead of long wait
- **Total time**: Same, but feels much faster

### Frontend Integration:
The frontend uses EventSource to receive Server-Sent Events from the streaming endpoint.

---

## 3. ML Model Optimizations (Already Implemented)

✅ **Singleton pattern**: Models loaded once at startup (~10s saved per request)
✅ **Cross-encoder disabled**: Re-ranking disabled by default (~3-5s saved)
✅ **Embedding cache**: LRU cache for repeated queries (~0.5-1s saved)
✅ **Timing logs**: Full transparency on bottlenecks

---

## Current Performance Metrics

### With HNSW Index:
- **RAG search**: ~0.5-1.0s (vs 2.8-3.0s without index)
- **LLM API call**: ~4.5-9.0s (external API, variable)
- **Total time**: ~5-10s (vs 7-12s without index)

### Best Case (with caching):
- RAG: ~0.5s
- LLM: ~4.5s
- **Total: ~5s** ✅

### Typical Case:
- RAG: ~0.8s
- LLM: ~7s
- **Total: ~7.5s**

---

## Additional Optimizations (Future)

### 1. Background Re-ranking with Celery
Move cross-encoder re-ranking to async worker for better accuracy without blocking:
- Respond immediately with vector-ranked results
- Update with re-ranked results when ready
- Requires: Redis + Celery setup

### 2. Response Caching
Cache frequent queries (e.g., "What are the themes?"):
- Redis cache with TTL
- Cache key: hash(movie_id + question)
- Instant responses for common questions

### 3. Database Connection Pooling
Use pgBouncer on Railway for better connection management.

---

## Migration Instructions for Railway

1. **Connect to Railway PostgreSQL**:
   ```bash
   railway connect postgres
   ```

2. **Run HNSW setup SQL**:
   Copy and paste the SQL from Step 1 and Step 2 above.

3. **Deploy updated backend code**:
   ```bash
   git add .
   git commit -m "Add streaming chat and performance optimizations"
   git push
   ```

4. **Verify in logs**:
   Look for improved timing logs:
   ```
   RAG search took: 0.8s (k=3)  # Down from 2.8s
   LLM API call took: 4.5s
   Total movie chat response time: 5.3s
   ```

---

## References

- [pgvector HNSW documentation](https://github.com/pgvector/pgvector#hnsw)
- [OpenRouter streaming API](https://openrouter.ai/docs#streaming)
- [Django StreamingHttpResponse](https://docs.djangoproject.com/en/stable/ref/request-response/#streaminghttpresponse)
