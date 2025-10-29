from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from movies.models import Movie
from reports.models import MovieSection


class Command(BaseCommand):
    help = 'List movies and their report status'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--incomplete',
            action='store_true',
            help='Show only movies with incomplete reports (< 6 sections)'
        )
        parser.add_argument(
            '--no-embeddings',
            action='store_true',
            help='Show only movies with sections missing embeddings'
        )
        parser.add_argument(
            '--complete',
            action='store_true',
            help='Show only movies with complete reports (6 sections)'
        )
        parser.add_argument(
            '--export-csv',
            type=str,
            help='Export results to CSV file'
        )
    
    def handle(self, *args, **options):
        # Get all movies with section counts
        movies = Movie.objects.annotate(
            total_sections=Count('sections'),
            sections_with_embeddings=Count(
                'sections',
                filter=Q(sections__embedding__isnull=False)
            )
        ).order_by('id')
        
        # Apply filters
        if options['incomplete']:
            movies = movies.filter(total_sections__lt=6)
        
        if options['complete']:
            movies = movies.filter(total_sections=6)
        
        if options['no_embeddings']:
            movies = movies.filter(
                total_sections__gt=0,
                sections_with_embeddings__lt=Count('sections')
            )
        
        if not movies.exists():
            self.stdout.write(self.style.WARNING("No movies match the criteria"))
            return
        
        # Display header
        self.stdout.write("\n" + "="*80)
        self.stdout.write(f"{'ID':<5} {'Title':<40} {'Sections':<12} {'Embeddings':<12} {'Status'}")
        self.stdout.write("="*80)
        
        # Display each movie
        data_rows = []
        for movie in movies:
            status = self._get_status(movie.total_sections, movie.sections_with_embeddings)
            status_color = self._get_status_color(status)
            
            self.stdout.write(
                f"{movie.id:<5} "
                f"{movie.title[:38]:<40} "
                f"{movie.total_sections}/6{'':<8} "
                f"{movie.sections_with_embeddings}/{movie.total_sections}{'':<8} "
                f"{status_color(status)}"
            )
            
            # Collect data for CSV export
            if options['export_csv']:
                data_rows.append({
                    'id': movie.id,
                    'title': movie.title,
                    'year': movie.year,
                    'sections': movie.total_sections,
                    'embeddings': movie.sections_with_embeddings,
                    'status': status
                })
        
        # Display summary
        self.stdout.write("="*80)
        total_movies = movies.count()
        complete = movies.filter(total_sections=6).count()
        incomplete = movies.filter(total_sections__lt=6, total_sections__gt=0).count()
        no_reports = movies.filter(total_sections=0).count()
        
        self.stdout.write(f"\nTotal movies: {total_movies}")
        self.stdout.write(self.style.SUCCESS(f"Complete (6/6): {complete}"))
        self.stdout.write(self.style.WARNING(f"Incomplete: {incomplete}"))
        self.stdout.write(self.style.ERROR(f"No reports: {no_reports}"))
        
        # Export to CSV if requested
        if options['export_csv']:
            self._export_csv(options['export_csv'], data_rows)
    
    def _get_status(self, total, with_embeddings):
        if total == 0:
            return "No Reports"
        elif total < 8: 
            return "Incomplete"
        elif with_embeddings < total:
            return "Missing Embeddings"
        else:
            return "Complete"
    
    def _get_status_color(self, status):
        if status == "Complete":
            return self.style.SUCCESS
        elif status == "Missing Embeddings":
            return self.style.WARNING
        else:
            return self.style.ERROR
    
    def _export_csv(self, filename, data):
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        
        self.stdout.write(f"\n✓ Exported to {filename}")