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
        """Check if embedding exists in MongoDB (not PostgreSQL)"""
        from reports.mongodb_models import MovieSectionMongoDB

        mongo_doc = MovieSectionMongoDB.get_by_movie_and_type(
            movie_id=obj.movie_id,
            section_type=obj.section_type
        )

        if mongo_doc and mongo_doc.get('embedding'):
            return format_html(
                '<span style="color: green; font-weight: bold;">✅ MongoDB</span>'
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
        """Show embedding info from MongoDB"""
        from reports.mongodb_models import MovieSectionMongoDB

        mongo_doc = MovieSectionMongoDB.get_by_movie_and_type(
            movie_id=obj.movie_id,
            section_type=obj.section_type
        )

        if not mongo_doc or not mongo_doc.get('embedding'):
            return format_html('<p style="color: red;">No embedding in MongoDB</p>')

        try:
            embedding = mongo_doc['embedding']
            dim = len(embedding)
            sample = embedding[:5]
        except (TypeError, AttributeError):
            return format_html('<p style="color: gray;">Embedding format error</p>')

        html = f'<p><strong>Storage:</strong> MongoDB</p>'
        html += f'<p><strong>Dimensions:</strong> {dim}</p>'
        html += f'<p><strong>Sample values:</strong> {sample}...</p>'

        return format_html(html)

    embedding_info.short_description = 'Embedding Info'
    
    @admin.action(description='🔧 Regenerate embeddings for selected sections → MongoDB')
    def regenerate_embeddings(self, request, queryset):
        from services.mongodb_rag_service import MongoDBRAGService
        from reports.mongodb_models import MovieSectionMongoDB

        rag = MongoDBRAGService()
        success = 0
        failed = 0

        for section in queryset:
            try:
                # Generate embedding
                embedding = rag.generate_embedding(section.content)

                # Check if exists in MongoDB
                existing = MovieSectionMongoDB.get_by_movie_and_type(
                    movie_id=section.movie_id,
                    section_type=section.section_type
                )

                if existing:
                    # Update
                    MovieSectionMongoDB.update_embedding(
                        movie_id=section.movie_id,
                        section_type=section.section_type,
                        embedding=embedding
                    )
                else:
                    # Create
                    MovieSectionMongoDB.create(
                        movie_id=section.movie_id,
                        section_type=section.section_type,
                        embedding=embedding
                    )

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
                f'Successfully regenerated {success} embeddings in MongoDB',
                level=messages.SUCCESS
            )

        if failed > 0:
            self.message_user(
                request,
                f'Failed to generate {failed} embeddings',
                level=messages.ERROR
            )
    
    @admin.action(description='❌ Delete embeddings from MongoDB (keep content in PostgreSQL)')
    def delete_embeddings(self, request, queryset):
        from reports.mongodb_models import MovieSectionMongoDB

        count = 0
        for section in queryset:
            # Delete from MongoDB
            deleted = MovieSectionMongoDB.delete_by_movie_and_type(
                movie_id=section.movie_id,
                section_type=section.section_type
            )
            if deleted:
                count += 1

        self.message_user(
            request,
            f'Deleted {count} embeddings from MongoDB',
            level=messages.WARNING
        )
    
    @admin.action(description='🗑️ Delete selected sections permanently (PostgreSQL + MongoDB)')
    def delete_sections(self, request, queryset):
        from reports.mongodb_models import MovieSectionMongoDB

        count = queryset.count()
        mongo_deleted = 0

        # Delete embeddings from MongoDB first
        for section in queryset:
            deleted = MovieSectionMongoDB.delete_by_movie_and_type(
                movie_id=section.movie_id,
                section_type=section.section_type
            )
            if deleted:
                mongo_deleted += 1

        # Delete sections from PostgreSQL
        queryset.delete()

        self.message_user(
            request,
            f'Permanently deleted {count} sections from PostgreSQL and {mongo_deleted} embeddings from MongoDB',
            level=messages.ERROR
        )