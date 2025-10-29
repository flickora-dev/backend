# reports/management/commands/check_section_status.py

from django.core.management.base import BaseCommand
from django.db.models import Count, Q
from movies.models import Movie
from reports.models import MovieSection


class Command(BaseCommand):
    help = 'Check section structure status after migration'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed breakdown by section type'
        )
    
    def handle(self, *args, **options):
        self.stdout.write("="*70)
        self.stdout.write("SECTION STRUCTURE STATUS")
        self.stdout.write("="*70)
        
        # Overall statistics
        total_movies = Movie.objects.count()
        total_sections = MovieSection.objects.count()
        
        self.stdout.write(f"\n📊 OVERALL STATISTICS:")
        self.stdout.write(f"  Total movies: {total_movies}")
        self.stdout.write(f"  Total sections: {total_sections}")
        if total_movies > 0:
            self.stdout.write(f"  Average sections per movie: {total_sections/total_movies:.1f}/8")
        
        # Old vs New section types
        self.stdout.write(f"\n📋 SECTION TYPE ANALYSIS:")
        
        old_types = ['basic_info', 'cast_performances', 'character_analysis', 
                     'thematic_artistic', 'critical_reception', 'legacy_impact']
        new_types = ['production', 'plot_structure', 'cast_crew', 'characters',
                     'visual_technical', 'themes', 'reception', 'legacy']
        
        old_count = MovieSection.objects.filter(section_type__in=old_types).count()
        new_count = MovieSection.objects.filter(section_type__in=new_types).count()
        
        self.stdout.write(f"  Old format sections: {old_count}")
        if old_count > 0:
            self.stdout.write(self.style.WARNING(f"    ⚠️  Run migration: python manage.py migrate_section_structure --rename-only"))
        
        self.stdout.write(f"  New format sections: {new_count}")
        if new_count > 0:
            self.stdout.write(self.style.SUCCESS(f"    ✓ Migration completed"))
        
        # Section type distribution
        if options['detailed']:
            self.stdout.write(f"\n📊 DETAILED DISTRIBUTION:")
            section_counts = MovieSection.objects.values('section_type').annotate(
                count=Count('id')
            ).order_by('section_type')
            
            for item in section_counts:
                section_name = dict(MovieSection.SECTION_TYPES).get(
                    item['section_type'], 
                    item['section_type']
                )
                count = item['count']
                percentage = (count / total_movies * 100) if total_movies > 0 else 0
                
                # Color code based on coverage
                if percentage >= 90:
                    style = self.style.SUCCESS
                elif percentage >= 50:
                    style = self.style.WARNING
                else:
                    style = self.style.ERROR
                
                self.stdout.write(style(
                    f"  {section_name}: {count}/{total_movies} ({percentage:.0f}%)"
                ))
        
        # Movies by completion status
        self.stdout.write(f"\n📈 COMPLETION STATUS:")
        
        movies_annotated = Movie.objects.annotate(
            section_count=Count('sections'),
            sections_with_embeddings=Count(
                'sections',
                filter=Q(sections__embedding__isnull=False)
            )
        )
        
        complete = movies_annotated.filter(section_count=8).count()
        partial = movies_annotated.filter(section_count__gt=0, section_count__lt=8).count()
        empty = movies_annotated.filter(section_count=0).count()
        
        self.stdout.write(self.style.SUCCESS(f"  ✓ Complete (8/8): {complete} movies"))
        self.stdout.write(self.style.WARNING(f"  ⚠️  Partial (<8): {partial} movies"))
        self.stdout.write(self.style.ERROR(f"  ✗ Empty (0/8): {empty} movies"))
        
        # Embedding status
        total_with_embeddings = MovieSection.objects.filter(
            embedding__isnull=False
        ).count()
        embedding_percentage = (total_with_embeddings / total_sections * 100) if total_sections > 0 else 0
        
        self.stdout.write(f"\n🔢 EMBEDDING STATUS:")
        self.stdout.write(f"  Sections with embeddings: {total_with_embeddings}/{total_sections} ({embedding_percentage:.0f}%)")
        
        if embedding_percentage < 100:
            missing = total_sections - total_with_embeddings
            self.stdout.write(self.style.WARNING(
                f"  ⚠️  {missing} sections missing embeddings"
            ))
            self.stdout.write(f"     Run: python manage.py generate_embeddings")
        
        # Next steps
        self.stdout.write(f"\n📝 NEXT STEPS:")
        
        if old_count > 0:
            self.stdout.write(self.style.WARNING(
                "  1. Rename old sections: python manage.py migrate_section_structure --rename-only"
            ))
        
        if partial > 0 or empty > 0:
            self.stdout.write(self.style.WARNING(
                f"  2. Generate missing sections: python manage.py migrate_section_structure --generate-missing"
            ))
        
        if total_sections > total_with_embeddings:
            self.stdout.write(self.style.WARNING(
                f"  3. Generate embeddings: python manage.py generate_embeddings"
            ))
        
        if complete == total_movies and embedding_percentage == 100:
            self.stdout.write(self.style.SUCCESS(
                "  ✓ All done! System is ready."
            ))
        
        self.stdout.write("="*70)