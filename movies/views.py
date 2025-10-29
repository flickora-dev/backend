from django.shortcuts import render, get_object_or_404, redirect
from django.views.generic import ListView, DetailView, TemplateView, CreateView
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login
from django.db.models import Q, Count
from django.urls import reverse_lazy
from .models import Movie, Genre, MovieView
from .forms import RegisterForm, LoginForm

class HomeView(TemplateView):
    template_name = 'home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['trending_movies'] = Movie.objects.all().order_by('-created_at')[:8]
        
        if self.request.user.is_authenticated:
            recent_views = MovieView.objects.filter(user=self.request.user).select_related('movie')[:5]
            context['recently_viewed'] = [view.movie for view in recent_views]
        else:
            context['recently_viewed'] = []
        
        context['popular_questions'] = [
            "What are the best sci-fi movies of 2024?",
            "Recommend movies similar to Christopher Nolan films",
            "What's trending in horror movies right now?"
        ]
        return context

class MovieListView(ListView):
    model = Movie
    template_name = 'movies/list.html'
    context_object_name = 'movies'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Movie.objects.all().prefetch_related('sections', 'genres')
        
        search_query = self.request.GET.get('search', '')
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) |
                Q(director__icontains=search_query)
            )
        
        genre_filter = self.request.GET.get('genre')
        if genre_filter:
            queryset = queryset.filter(genres__tmdb_id=genre_filter).distinct()
        
        year_filter = self.request.GET.get('year')
        if year_filter and year_filter != 'all':
            queryset = queryset.filter(year=year_filter)
        
        return queryset.order_by('-year', 'title')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_query'] = self.request.GET.get('search', '')
        context['selected_genre'] = self.request.GET.get('genre', '')
        context['selected_year'] = self.request.GET.get('year', 'all')
        
        context['all_genres'] = Genre.objects.annotate(
            movie_count=Count('movies')
        ).filter(movie_count__gt=0).order_by('name')
        
        context['all_years'] = Movie.objects.values_list('year', flat=True).distinct().order_by('-year')
        
        return context

class MovieDetailView(DetailView):
    model = Movie
    template_name = 'movies/detail.html'
    context_object_name = 'movie'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['sections'] = self.object.sections.all().order_by('section_type')
        
        if self.request.user.is_authenticated:
            MovieView.objects.update_or_create(
                user=self.request.user,
                movie=self.object
            )
        
        return context


class RegisterView(CreateView):
    form_class = RegisterForm
    template_name = 'auth/register.html'
    success_url = reverse_lazy('home')
    
    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect(self.success_url)


class CustomLoginView(LoginView):
    form_class = LoginForm
    template_name = 'auth/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('home')


class CustomLogoutView(LogoutView):
    next_page = reverse_lazy('home')