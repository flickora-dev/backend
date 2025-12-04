from django.contrib import admin
from django.utils.html import format_html
from django.contrib import messages
from .models import MovieSection
from services.mongodb_service import get_mongodb_service


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
        mongodb = get_mongodb_service()
        has_embedding = mongodb.get_embedding(obj.id) is not None

        if has_embedding:
            return format_html(
                '<span style="color: green; font-weight: bold;">✅ Yes (MongoDB)</span>'
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
        mongodb = get_mongodb_service()
        mongo_doc = mongodb.get_embedding(obj.id)

        if not mongo_doc:
            return format_html('<p style="color: red;">No embedding in MongoDB</p>')

        try:
            dim = mongo_doc.get('dimensions', 0)
            embedding = mongo_doc.get('embedding', [])
            sample = embedding[:5] if len(embedding) >= 5 else embedding
        except (TypeError, AttributeError, KeyError):
            return format_html('<p style="color: gray;">Embedding data error</p>')

        html = f'<p><strong>Storage:</strong> MongoDB Atlas</p>'
        html += f'<p><strong>Dimensions:</strong> {dim}</p>'
        html += f'<p><strong>Sample values:</strong> {sample}...</p>'

        return format_html(html)

    embedding_info.short_description = 'Embedding Info'
    
    @admin.action(description='🔧 Regenerate embeddings for selected sections')
    def regenerate_embeddings(self, request, queryset):
        from services.rag_service import RAGService

        rag = RAGService()
        mongodb = get_mongodb_service()
        success = 0
        failed = 0

        for section in queryset:
            try:
                embedding = rag.generate_embedding(section.content)

                # Store in MongoDB
                success_stored = mongodb.store_embedding(
                    section_id=section.id,
                    movie_id=section.movie_id,
                    section_type=section.section_type,
                    embedding=embedding.tolist(),
                    metadata={
                        'movie_title': section.movie.title,
                        'section_type_display': section.get_section_type_display(),
                        'word_count': section.word_count,
                        'content_preview': section.content[:200]
                    }
                )

                if success_stored:
                    success += 1
                else:
                    failed += 1
                    self.message_user(
                        request,
                        f'Failed to store in MongoDB for {section.movie.title} - {section.section_type}',
                        level=messages.ERROR
                    )
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
                f'Successfully regenerated {success} embeddings in MongoDB',
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
        mongodb = get_mongodb_service()
        count = 0

        for section in queryset:
            if mongodb.delete_embedding(section.id):
                count += 1

        self.message_user(
            request,
            f'Deleted {count} embeddings from MongoDB',
            level=messages.WARNING
        )

    @admin.action(description='🗑️ Delete selected sections permanently')
    def delete_sections(self, request, queryset):
        mongodb = get_mongodb_service()
        count = queryset.count()

        # Delete embeddings from MongoDB
        for section in queryset:
            mongodb.delete_embedding(section.id)

        # Delete sections from PostgreSQL
        queryset.delete()

        self.message_user(
            request,
            f'Permanently deleted {count} sections and their embeddings',
            level=messages.ERROR
        )