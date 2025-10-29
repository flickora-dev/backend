from django.core.management.base import BaseCommand
from movies.models import Movie
from reports.models import MovieSection


class Command(BaseCommand):
    help = 'Clean up movie reports and embeddings with various options'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--movie-id',
            type=int,
            help='Delete reports for specific movie ID'
        )
        parser.add_argument(
            '--movie-title',
            type=str,
            help='Delete reports for movie by title (partial match)'
        )
        parser.add_argument(
            '--section-type',
            type=str,
            choices=[
                'production',
                'plot_structure',
                'cast_crew',
                'characters',
                'visual_technical',
                'themes',
                'reception',
                'legacy'
            ],
            help='Delete only specific section type'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Delete ALL reports (use with caution!)'
        )
        parser.add_argument(
            '--incomplete',
            action='store_true',
            help='Delete reports for movies with incomplete sections (< 6)'
        )
        parser.add_argument(
            '--no-embeddings',
            action='store_true',
            help='Delete only sections without embeddings'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )
        parser.add_argument(
            '--confirm',
            action='store_true',
            help='Skip confirmation prompt (use in scripts)'
        )
    
    def handle(self, *args, **options):
        # Build queryset based on options
        queryset = MovieSection.objects.all()
        
        if options['movie_id']:
            queryset = queryset.filter(movie_id=options['movie_id'])
            movie = Movie.objects.get(id=options['movie_id'])
            self.stdout.write(f"Filtering by movie: {movie.title}")
        
        if options['movie_title']:
            queryset = queryset.filter(movie__title__icontains=options['movie_title'])
            count = queryset.values('movie').distinct().count()
            self.stdout.write(f"Found {count} movies matching '{options['movie_title']}'")
        
        if options['section_type']:
            queryset = queryset.filter(section_type=options['section_type'])
            self.stdout.write(f"Filtering by section type: {options['section_type']}")
        
        if options['incomplete']:
            # Find movies with < 6 sections
            from django.db.models import Count
            incomplete_movies = Movie.objects.annotate(
                section_count=Count('sections')
            ).filter(section_count__lt=8)
            queryset = queryset.filter(movie__in=incomplete_movies)
            self.stdout.write(f"Found {incomplete_movies.count()} movies with incomplete reports")
        
        if options['no_embeddings']:
            queryset = queryset.filter(embedding__isnull=True)
            self.stdout.write("Filtering sections without embeddings")
        
        # Get statistics
        total_sections = queryset.count()
        affected_movies = queryset.values('movie').distinct().count()
        sections_with_embeddings = queryset.filter(embedding__isnull=False).count()
        
        if total_sections == 0:
            self.stdout.write(self.style.WARNING("No sections match the criteria"))
            return
        
        # Display summary
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.WARNING("DELETION SUMMARY"))
        self.stdout.write("="*60)
        self.stdout.write(f"Sections to delete: {total_sections}")
        self.stdout.write(f"Affected movies: {affected_movies}")
        self.stdout.write(f"Sections with embeddings: {sections_with_embeddings}")
        
        # Show affected movies
        if affected_movies <= 20:
            self.stdout.write("\nAffected movies:")
            for movie in queryset.values('movie__id', 'movie__title').distinct():
                section_count = queryset.filter(movie_id=movie['movie__id']).count()
                self.stdout.write(f"  - {movie['movie__title']} (ID: {movie['movie__id']}) - {section_count} sections")
        
        # Dry run mode
        if options['dry_run']:
            self.stdout.write("\n" + self.style.SUCCESS("DRY RUN - No data was deleted"))
            return
        
        # Confirmation
        if not options['confirm'] and not options['all']:
            confirm = input("\nDo you want to proceed? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                self.stdout.write(self.style.ERROR("Cancelled"))
                return
        
        if options['all'] and not options['confirm']:
            confirm = input("\n⚠️  WARNING: This will delete ALL reports! Type 'DELETE ALL' to confirm: ")
            if confirm != 'DELETE ALL':
                self.stdout.write(self.style.ERROR("Cancelled"))
                return
        
        # Perform deletion
        deleted_count = queryset.delete()[0]
        
        self.stdout.write("\n" + "="*60)
        self.stdout.write(self.style.SUCCESS(f"✓ Successfully deleted {deleted_count} sections"))
        self.stdout.write("="*60)
        
        # Show remaining stats
        remaining_sections = MovieSection.objects.count()
        remaining_with_embeddings = MovieSection.objects.filter(embedding__isnull=False).count()
        self.stdout.write(f"\nRemaining sections: {remaining_sections}")
        self.stdout.write(f"Sections with embeddings: {remaining_with_embeddings}")