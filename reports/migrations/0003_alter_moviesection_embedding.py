from django.db import migrations
import pgvector.django


class Migration(migrations.Migration):

    dependencies = [
        ("reports", "0002_moviesection_embedding"),
    ]

    operations = [
        migrations.RunSQL(
            sql='CREATE EXTENSION IF NOT EXISTS vector;',
            reverse_sql='DROP EXTENSION IF EXISTS vector CASCADE;'
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