from django.db import models
from django.contrib.auth.models import User

class Genre(models.Model):
    tmdb_id = models.IntegerField(unique=True)
    name = models.CharField(max_length=100)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name

class Movie(models.Model):
    tmdb_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    year = models.IntegerField()
    director = models.CharField(max_length=255, blank=True)
    genres = models.ManyToManyField(Genre, related_name='movies')
    imdb_rating = models.DecimalField(max_digits=3, decimal_places=1, null=True, blank=True)
    plot_summary = models.TextField(blank=True)
    poster_url = models.URLField(blank=True)
    backdrop_url = models.URLField(blank=True)
    runtime = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-year', 'title']
    
    def __str__(self):
        return f"{self.title} ({self.year})"
    
    @property
    def genre_list(self):
        return ", ".join([g.name for g in self.genres.all()])


class MovieView(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='movie_views')
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name='views')
    viewed_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-viewed_at']
        unique_together = ['user', 'movie']
    
    def __str__(self):
        return f"{self.user.username} viewed {self.movie.title}"