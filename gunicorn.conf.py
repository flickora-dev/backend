"""
Gunicorn configuration for flickora backend
Optimized for async performance with uvicorn workers
"""
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Worker processes
# Use 2-4 workers for better concurrency
workers = int(os.getenv('GUNICORN_WORKERS', 2))

# Worker class - use uvicorn for async support
# This improves performance for I/O-bound operations (database, API calls)
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout - increase for long RAG queries
timeout = 120

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Preload app for faster worker spawn times
preload_app = True

# Graceful timeout
graceful_timeout = 30

# Keep alive
keepalive = 5

# Maximum requests per worker before restart (prevent memory leaks)
max_requests = 1000
max_requests_jitter = 50
