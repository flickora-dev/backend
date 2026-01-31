import django_filters
from django.db.models import Count
from movies.models import Movie


class GenresAllFilter(django_filters.BaseInFilter):
    """Filter that requires ALL genres to match (AND logic)"""

    def filter(self, qs, value):
        if not value:
            return qs

        # Filter movies that have ALL selected genres
        for genre_id in value:
            qs = qs.filter(genres__tmdb_id=genre_id)

        # Remove duplicates
        return qs.distinct()


class MovieFilter(django_filters.FilterSet):
    """Custom filter for movies supporting multiple values"""

    # Single value filters
    year = django_filters.NumberFilter(field_name='year')
    genres__tmdb_id = django_filters.NumberFilter(field_name='genres__tmdb_id')

    # Year range filters
    year_from = django_filters.NumberFilter(field_name='year', lookup_expr='gte')
    year_to = django_filters.NumberFilter(field_name='year', lookup_expr='lte')

    # Multiple genres with AND logic (movie must have ALL selected genres)
    genres__tmdb_id__in = GenresAllFilter(field_name='genres__tmdb_id')

    class Meta:
        model = Movie
        fields = ['year', 'genres__tmdb_id', 'year_from', 'year_to', 'genres__tmdb_id__in']
