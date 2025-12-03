# reports/management/commands/test_mongodb_search.py
"""
Django management command to test MongoDB vector search.

Usage:
    python manage.py test_mongodb_search [--query "your query"]
"""
from django.core.management.base import BaseCommand
from services.mongodb_rag_service import MongoDBRAGService
import time


class Command(BaseCommand):
    help = 'Test MongoDB vector search functionality'

    def add_arguments(self, parser):
        parser.add_argument(
            '--query',
            type=str,
            default='What are the main themes in this movie?',
            help='Search query to test (default: "What are the main themes in this movie?")'
        )
        parser.add_argument(
            '--k',
            type=int,
            default=5,
            help='Number of results to return (default: 5)'
        )

    def handle(self, *args, **options):
        query = options['query']
        k = options['k']

        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS('MongoDB Vector Search Test'))
        self.stdout.write(self.style.SUCCESS('=' * 80))

        self.stdout.write(f'\nQuery: "{query}"')
        self.stdout.write(f'Retrieving top {k} results\n')

        # Initialize RAG service
        self.stdout.write('Initializing MongoDB RAG Service...')
        try:
            rag_service = MongoDBRAGService()
            self.stdout.write(self.style.SUCCESS('✅ Service initialized\n'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Failed to initialize service: {e}'))
            return

        # Perform search
        self.stdout.write('Performing vector search...')
        start_time = time.time()

        try:
            results = rag_service.search_with_scores(query=query, k=k)
            elapsed_time = time.time() - start_time

            self.stdout.write(self.style.SUCCESS(f'✅ Search completed in {elapsed_time:.3f}s\n'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Search failed: {e}'))
            import traceback
            traceback.print_exc()
            return

        # Display results
        if not results:
            self.stdout.write(self.style.WARNING('No results found'))
            self.stdout.write(self.style.WARNING('\nPossible reasons:'))
            self.stdout.write('  1. No embeddings in MongoDB collection')
            self.stdout.write('  2. Vector search index not created')
            self.stdout.write('  3. Minimum similarity threshold too high')
            self.stdout.write('\nTry running:')
            self.stdout.write('  python manage.py migrate_to_mongodb')
            self.stdout.write('  python manage.py create_mongodb_indexes')
            return

        self.stdout.write('=' * 80)
        self.stdout.write(self.style.SUCCESS(f'FOUND {len(results)} RESULTS'))
        self.stdout.write('=' * 80)

        for i, result in enumerate(results, 1):
            self.stdout.write(f'\n{i}. {result["movie_title"]} - {result["section_type"]}')
            self.stdout.write(f'   Similarity: {result["similarity"]:.4f}')
            self.stdout.write(f'   Weighted Score: {result["weighted_score"]:.4f}')
            self.stdout.write(f'   Word Count: {result["word_count"]}')
            self.stdout.write(f'   Content Preview: {result["content"][:200]}...\n')

        # Performance summary
        self.stdout.write('=' * 80)
        self.stdout.write(self.style.SUCCESS('PERFORMANCE SUMMARY'))
        self.stdout.write('=' * 80)
        self.stdout.write(f'Query time: {elapsed_time:.3f}s')
        self.stdout.write(f'Results: {len(results)}')
        self.stdout.write(f'Avg similarity: {sum(r["similarity"] for r in results) / len(results):.4f}')

        self.stdout.write('\n' + self.style.SUCCESS('✅ Test completed successfully!'))
