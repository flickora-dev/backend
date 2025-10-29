from django.contrib import admin
from django.db.models import Count, Q
from django.utils.html import format_html
from django.shortcuts import redirect
from django.urls import path
from django.contrib import messages
from .models import Movie, Genre, MovieView


@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'tmdb_id', 'movie_count']
    search_fields = ['name']
    ordering = ['name']
    
    def movie_count(self, obj):
        count = obj.movies.count()
        return format_html(
            '<span style="color: {};">{} movies</span>',
            'green' if count > 0 else 'gray',
            count
        )
    movie_count.short_description = 'Movies'


@admin.register(MovieView)
class MovieViewAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'movie', 'viewed_at']
    list_filter = ['viewed_at', 'user']
    search_fields = ['user__username', 'movie__title']
    readonly_fields = ['viewed_at']
    ordering = ['-viewed_at']


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = [
        'id', 
        'title', 
        'year', 
        'director', 
        'genre_display',
        'imdb_rating',
        'view_count',
        'section_status',
        'embedding_status'
    ]
    list_filter = ['year', 'genres']
    search_fields = ['title', 'director']
    readonly_fields = ['created_at', 'updated_at', 'report_details', 'view_stats']
    filter_horizontal = ['genres']
    ordering = ['id']
    actions = [
        'generate_reports_action',
        'delete_reports_action',
        'regenerate_embeddings_action',
        'delete_embeddings_action'
    ]
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        queryset = queryset.prefetch_related('genres').annotate(
            total_sections=Count('sections'),
            sections_with_embeddings=Count(
                'sections',
                filter=Q(sections__embedding__isnull=False)
            ),
            total_views=Count('views')
        )
        return queryset
    
    def genre_display(self, obj):
        genres = obj.genres.all()[:3]
        if not genres:
            return format_html('<span style="color: gray;">-</span>')
        
        genre_names = ', '.join([g.name for g in genres])
        if obj.genres.count() > 3:
            genre_names += f' (+{obj.genres.count() - 3})'
        
        return format_html('<span>{}</span>', genre_names)
    genre_display.short_description = 'Genres'
    
    def view_count(self, obj):
        count = obj.total_views if hasattr(obj, 'total_views') else obj.views.count()
        return format_html(
            '<span style="color: {};">{} views</span>',
            'green' if count > 0 else 'gray',
            count
        )
    view_count.short_description = 'Views'
    view_count.admin_order_field = 'total_views'
    
    def section_status(self, obj):
        total = obj.total_sections if hasattr(obj, 'total_sections') else obj.sections.count()
        
        if total == 0:
            color = 'red'
            icon = '❌'
        elif total < 8:  
            color = 'orange'
            icon = '⚠️'
        else:
            color = 'green'
            icon = '✅'
        
        return format_html(
            '<span style="color: {};">{} {}/8</span>',  
            color, icon, total
        )
    section_status.short_description = 'Reports'
    section_status.admin_order_field = 'total_sections'
    
    def embedding_status(self, obj):
        total = obj.total_sections if hasattr(obj, 'total_sections') else obj.sections.count()
        with_emb = obj.sections_with_embeddings if hasattr(obj, 'sections_with_embeddings') else obj.sections.filter(embedding__isnull=False).count()
        
        if total == 0:
            return format_html('<span style="color: gray;">-</span>')
        elif with_emb == total:
            return format_html('<span style="color: green;">✅ {}/{}</span>', with_emb, total)
        elif with_emb > 0:
            return format_html('<span style="color: orange;">⚠️ {}/{}</span>', with_emb, total)
        else:
            return format_html('<span style="color: red;">❌ 0/{}</span>', total)
    
    embedding_status.short_description = 'Embeddings'
    
    def view_stats(self, obj):
        views = obj.views.all().select_related('user')[:10]
        
        if not views:
            return format_html('<p style="color: gray;">No views yet</p>')
        
        html = '<table style="width: 100%; border-collapse: collapse;">'
        html += '<tr style="background: #f0f0f0;"><th>User</th><th>Viewed At</th></tr>'
        
        for view in views:
            html += f'<tr><td>{view.user.username}</td><td>{view.viewed_at.strftime("%Y-%m-%d %H:%M")}</td></tr>'
        
        html += '</table>'
        
        total = obj.views.count()
        if total > 10:
            html += f'<p style="margin-top: 10px;">... and {total - 10} more views</p>'
        
        return format_html(html)
    
    view_stats.short_description = 'Recent Views'
    
    def report_details(self, obj):
        sections = obj.sections.all()
        
        if not sections:
            return format_html('<p style="color: gray;">No reports generated</p>')
        
        html = '<table style="width: 100%; border-collapse: collapse;">'
        html += '<tr style="background: #f0f0f0;"><th>Section Type</th><th>Words</th><th>Embedding</th></tr>'
        
        for section in sections:
            has_emb = section.embedding is not None and len(section.embedding) > 0
            emb_status = '✅' if has_emb else '❌'
            html += f'<tr><td>{section.get_section_type_display()}</td><td>{section.word_count}</td><td>{emb_status}</td></tr>'
        
        html += '</table>'
        return format_html(html)
    
    report_details.short_description = 'Report Details'
    
    @admin.action(description='🔄 Generate reports for selected movies')
    def generate_reports_action(self, request, queryset):
        from django.core.management import call_command
        import io
        
        count = queryset.count()
        
        for movie in queryset:
            try:
                out = io.StringIO()
                call_command('generate_reports', movie_id=movie.id, stdout=out)
                
            except Exception as e:
                self.message_user(
                    request,
                    f'Error generating reports for {movie.title}: {str(e)}',
                    level=messages.ERROR
                )
                continue
        
        self.message_user(
            request,
            f'Started report generation for {count} movie(s). This may take a while...',
            level=messages.SUCCESS
        )
    
    @admin.action(description='🗑️ Delete reports for selected movies')
    def delete_reports_action(self, request, queryset):
        total_deleted = 0
        
        for movie in queryset:
            count = movie.sections.count()
            movie.sections.all().delete()
            total_deleted += count
        
        self.message_user(
            request,
            f'Deleted {total_deleted} report sections from {queryset.count()} movie(s)',
            level=messages.WARNING
        )
    
    @admin.action(description='🔧 Regenerate embeddings for selected movies')
    def regenerate_embeddings_action(self, request, queryset):
        from services.rag_service import RAGService
        
        rag = RAGService()
        total_processed = 0
        
        for movie in queryset:
            sections = movie.sections.all()
            
            for section in sections:
                try:
                    embedding = rag.generate_embedding(section.content)
                    section.embedding = embedding
                    section.save(update_fields=['embedding'])
                    total_processed += 1
                except Exception as e:
                    self.message_user(
                        request,
                        f'Error generating embedding for {movie.title} - {section.section_type}: {str(e)}',
                        level=messages.ERROR
                    )
        
        self.message_user(
            request,
            f'Generated {total_processed} embeddings for {queryset.count()} movie(s)',
            level=messages.SUCCESS
        )
    
    @admin.action(description='❌ Delete embeddings (keep reports)')
    def delete_embeddings_action(self, request, queryset):
        total_deleted = 0
        
        for movie in queryset:
            sections = movie.sections.all()
            for section in sections:
                if section.embedding is not None and len(section.embedding) > 0:
                    section.embedding = None
                    section.save(update_fields=['embedding'])
                    total_deleted += 1
        
        self.message_user(
            request,
            f'Deleted {total_deleted} embeddings from {queryset.count()} movie(s)',
            level=messages.WARNING
        )