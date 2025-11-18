from django.db import migrations, connection
import pgvector.django


def create_vector_extension(apps, schema_editor):
    """Create pgvector extension only if using PostgreSQL"""
    if connection.vendor == 'postgresql':
        schema_editor.execute('CREATE EXTENSION IF NOT EXISTS vector;')


def drop_vector_extension(apps, schema_editor):
    """Drop pgvector extension only if using PostgreSQL"""
    if connection.vendor == 'postgresql':
        schema_editor.execute('DROP EXTENSION IF EXISTS vector CASCADE;')


class Migration(migrations.Migration):

    dependencies = [
        ("reports", "0002_moviesection_embedding"),
    ]

    operations = [
        # Only create extension if using PostgreSQL
        migrations.RunPython(
            create_vector_extension,
            reverse_code=drop_vector_extension,
        ),

        migrations.RemoveField(
            model_name='moviesection',
            name='embedding',
        ),

        migrations.AddField(
            model_name='moviesection',
            name='embedding',
            field=pgvector.django.VectorField(dimensions=384, null=True, blank=True),
        ),
    ]