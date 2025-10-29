from django.db import models
from movies.models import Movie
from pgvector.django import VectorField

class MovieSection(models.Model):
    SECTION_TYPES = [
        ('production', 'Production & Release'),
        ('plot_structure', 'Plot & Structure'),
        ('cast_crew', 'Cast & Crew'),
        ('characters', 'Characters & Relationships'),
        ('visual_technical', 'Visual & Technical Mastery'),
        ('themes', 'Themes & Symbolism'),
        ('reception', 'Critical Reception & Analysis'),
        ('legacy', 'Cultural Impact & Legacy'),
    ]
    
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='sections')
    section_type = models.CharField(max_length=50, choices=SECTION_TYPES)
    content = models.TextField()
    word_count = models.IntegerField(default=0)
    key_topics = models.JSONField(default=list, blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=384, null=True, blank=True)
    
    class Meta:
        unique_together = ['movie', 'section_type']
        ordering = ['movie', 'section_type']
        indexes = [
            models.Index(fields=['section_type']),
            models.Index(fields=['generated_at']),
        ]
    
    def __str__(self):
        return f"{self.movie.title} - {self.get_section_type_display()}"
    
    def save(self, *args, **kwargs):
        if self.content:
            self.word_count = len(self.content.split())
        super().save(*args, **kwargs)
    
    @property
    def tier(self):
        tier_mapping = {
            'production': 1, 'plot_structure': 1, 'cast_crew': 1,
            'characters': 2, 'visual_technical': 2, 'themes': 2,
            'reception': 3, 'legacy': 3,
        }
        return tier_mapping.get(self.section_type, 0)
    
    @property
    def target_word_count(self):
        targets = {
            'production': 400,
            'plot_structure': 600,
            'cast_crew': 400,
            'characters': 500,
            'visual_technical': 500,
            'themes': 500,
            'reception': 400,
            'legacy': 400,
        }
        return targets.get(self.section_type, 500)