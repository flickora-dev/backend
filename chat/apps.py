from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):
        """
        Preload ML models on Django startup to avoid loading on every request
        """
        # Only load in worker processes, not during migrations
        import sys
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv[0] or 'uwsgi' in sys.argv[0]:
            try:
                logger.info("Preloading ML models for RAG services...")
                from services.optimized_rag_service import _ModelSingleton

                # Force singleton initialization
                singleton = _ModelSingleton()
                logger.info("ML models preloaded successfully")
            except Exception as e:
                logger.error(f"Error preloading models: {e}")
