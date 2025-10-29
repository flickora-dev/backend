from django.contrib import admin
from django.utils.html import format_html
from django.contrib import messages
from .models import MovieSection


@admin.register(MovieSection)
class MovieSectionAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'movie', 
        'section_type', 
        'word_count',
        'embedding_status',
        'generated_at'
    ]
    list_filter = ['section_type', 'generated_at', 'movie']
    search_fields = ['movie__title', 'content']
    readonly_fields = ['word_count', 'generated_at', 'content_preview', 'embedding_info']
    ordering = ['-generated_at']
    actions = [
        'regenerate_embeddings',
        'delete_embeddings',
        'delete_sections'
    ]
    
    def embedding_status(self, obj):
        if obj.embedding is not None and len(obj.embedding) > 0:
            return format_html(
                '<span style="color: green; font-weight: bold;">✅ Yes</span>'
            )
        return format_html(
            '<span style="color: red; font-weight: bold;">❌ No</span>'
        )
    
    embedding_status.short_description = 'Has Embedding'
    
    def content_preview(self, obj):
        preview = obj.content[:500] + '...' if len(obj.content) > 500 else obj.content
        return format_html('<p style="white-space: pre-wrap;">{}</p>', preview)
    
    content_preview.short_description = 'Content Preview'
    
    def embedding_info(self, obj):
        if obj.embedding is None or len(obj.embedding) == 0:
            return format_html('<p style="color: red;">No embedding generated</p>')
        
        try:
            dim = len(obj.embedding)
            sample = obj.embedding[:5].tolist() if hasattr(obj.embedding, 'tolist') else list(obj.embedding[:5])
        except (TypeError, AttributeError):
            return format_html('<p style="color: gray;">Embedding format: {}</p>', type(obj.embedding).__name__)
        
        html = f'<p><strong>Dimensions:</strong> {dim}</p>'
        html += f'<p><strong>Sample values:</strong> {sample}...</p>'
        
        return format_html(html)
    
    embedding_info.short_description = 'Embedding Info'
    
    @admin.action(description='🔧 Regenerate embeddings for selected sections')
    def regenerate_embeddings(self, request, queryset):
        from services.rag_service import RAGService
        
        rag = RAGService()
        success = 0
        failed = 0
        
        for section in queryset:
            try:
                embedding = rag.generate_embedding(section.content)
                section.embedding = embedding
                section.save(update_fields=['embedding'])
                success += 1
            except Exception as e:
                failed += 1
                self.message_user(
                    request,
                    f'Failed for {section.movie.title} - {section.section_type}: {str(e)}',
                    level=messages.ERROR
                )
        
        if success > 0:
            self.message_user(
                request,
                f'Successfully regenerated {success} embeddings',
                level=messages.SUCCESS
            )
        
        if failed > 0:
            self.message_user(
                request,
                f'Failed to generate {failed} embeddings',
                level=messages.ERROR
            )
    
    @admin.action(description='❌ Delete embeddings (keep content)')
    def delete_embeddings(self, request, queryset):
        count = 0
        for section in queryset:
            if section.embedding is not None and len(section.embedding) > 0:
                section.embedding = None
                section.save(update_fields=['embedding'])
                count += 1
        
        self.message_user(
            request,
            f'Deleted {count} embeddings',
            level=messages.WARNING
        )
    
    @admin.action(description='🗑️ Delete selected sections permanently')
    def delete_sections(self, request, queryset):
        count = queryset.count()
        queryset.delete()
        
        self.message_user(
            request,
            f'Permanently deleted {count} sections',
            level=messages.ERROR
        )