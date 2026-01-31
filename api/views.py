from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db import models
from movies.models import Movie, Genre
from reports.models import MovieSection
from services.tmdb_service import TMDBService
from services.openrouter_service import OpenRouterService
from services.rag_service import RAGService
import json
import logging
from django.http import JsonResponse

def health(request):
    return JsonResponse({"status": "ok"})

logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def import_movie(request):
    try:
        data = json.loads(request.body)
        tmdb_id = data.get('tmdb_id')
        
        if not tmdb_id:
            return JsonResponse({'error': 'tmdb_id required'}, status=400)
        
        tmdb = TMDBService()
        movie_data = tmdb.get_movie_details(tmdb_id)
        
        if not movie_data:
            return JsonResponse({'error': 'Movie not found in TMDB'}, status=404)
        
        movie, created = Movie.objects.get_or_create(
            tmdb_id=tmdb_id,
            defaults={
                'title': movie_data['title'],
                'year': int(movie_data['release_date'][:4]) if movie_data.get('release_date') else 2024,
                'director': get_director(movie_data),
                'plot_summary': movie_data.get('overview', ''),
                'runtime': movie_data.get('runtime'),
                'imdb_rating': movie_data.get('vote_average'),
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}" if movie_data.get('poster_path') else '',
                'backdrop_url': f"https://image.tmdb.org/t/p/w1280{movie_data['backdrop_path']}" if movie_data.get('backdrop_path') else '',
            }
        )
        
        for genre_data in movie_data.get('genres', []):
            genre, _ = Genre.objects.get_or_create(
                tmdb_id=genre_data['id'],
                defaults={'name': genre_data['name']}
            )
            movie.genres.add(genre)
        
        return JsonResponse({
            'success': True,
            'created': created,
            'movie': {
                'id': movie.id,
                'title': movie.title,
                'year': movie.year,
                'tmdb_id': movie.tmdb_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error importing movie: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_section(request):
    try:
        data = json.loads(request.body)
        movie_id = data.get('movie_id')
        section_type = data.get('section_type')
        
        if not movie_id or not section_type:
            return JsonResponse({'error': 'movie_id and section_type required'}, status=400)
        
        valid_types = [choice[0] for choice in MovieSection.SECTION_TYPES]
        if section_type not in valid_types:
            return JsonResponse({
                'error': f'Invalid section_type. Valid types: {", ".join(valid_types)}'
            }, status=400)
        
        movie = Movie.objects.get(id=movie_id)
        
        if MovieSection.objects.filter(movie=movie, section_type=section_type).exists():
            return JsonResponse({'error': 'Section already exists'}, status=400)
        
        openrouter = OpenRouterService()
        movie_data = {
            'title': movie.title,
            'year': movie.year,
            'director': movie.director,
            'genres': ', '.join([g.name for g in movie.genres.all()]),
            'plot_summary': movie.plot_summary
        }
        
        content = openrouter.generate_movie_section(movie_data, section_type)
        
        if not content:
            return JsonResponse({'error': 'Failed to generate content'}, status=500)

        section = MovieSection.objects.create(
            movie=movie,
            section_type=section_type,
            content=content
        )
        
        return JsonResponse({
            'success': True,
            'section': {
                'id': section.id,
                'section_type': section.section_type,
                'word_count': section.word_count,
                'has_embedding': False,  # Zawsze False - generuj osobno przez /api/generate-embedding/
                'movie_id': movie.id
            }
        })
        
    except Movie.DoesNotExist:
        return JsonResponse({'error': 'Movie not found'}, status=404)
    except Exception as e:
        logger.error(f"Error generating section: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def generate_embedding(request):
    try:
        from services.mongodb_service import get_mongodb_service

        data = json.loads(request.body)
        section_id = data.get('section_id')

        if not section_id:
            return JsonResponse({'error': 'section_id required'}, status=400)

        section = MovieSection.objects.get(id=section_id)

        # Check if embedding already exists in MongoDB
        mongodb = get_mongodb_service()
        existing = mongodb.get_embedding(section_id)
        if existing:
            return JsonResponse({'error': 'Embedding already exists'}, status=400)

        rag = RAGService()

        try:
            logger.info(f"Generating embedding for section {section_id}")
            embedding = rag.generate_embedding(section.content)

            # Store embedding in MongoDB
            success = mongodb.store_embedding(
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

            if not success:
                raise Exception("Failed to store embedding in MongoDB")

            logger.info(f"Successfully generated and stored embedding for section {section_id}")

            return JsonResponse({
                'success': True,
                'section_id': section.id,
                'embedding_dimensions': len(embedding) if embedding is not None else 0
            })

        except Exception as e:
            logger.error(f"Error generating embedding for section {section_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return JsonResponse({
                'error': f'Embedding generation failed: {str(e)}'
            }, status=500)

    except MovieSection.DoesNotExist:
        return JsonResponse({'error': 'Section not found'}, status=404)
    except Exception as e:
        logger.error(f"Error in generate_embedding endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JsonResponse({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def movie_status(request, movie_id):
    try:
        from services.mongodb_service import get_mongodb_service

        movie = Movie.objects.get(id=movie_id)
        sections = MovieSection.objects.filter(movie=movie)
        mongodb = get_mongodb_service()

        section_status = {}
        for section in sections:
            # Check if embedding exists in MongoDB
            mongo_doc = mongodb.get_embedding(section.id)
            has_embedding = mongo_doc is not None

            section_status[section.section_type] = {
                'exists': True,
                'word_count': section.word_count,
                'has_embedding': has_embedding
            }

        all_types = ['production', 'plot_structure', 'cast_crew', 'characters',
                     'visual_technical', 'themes', 'reception', 'legacy']

        for section_type in all_types:
            if section_type not in section_status:
                section_status[section_type] = {
                    'exists': False,
                    'word_count': 0,
                    'has_embedding': False
                }

        return JsonResponse({
            'movie_id': movie.id,
            'title': movie.title,
            'sections': section_status,
            'total_sections': sections.count(),
            'complete': sections.count() == 8
        })

    except Movie.DoesNotExist:
        return JsonResponse({'error': 'Movie not found'}, status=404)


@require_http_methods(["GET"])
def movies_without_reports(request):
    from services.mongodb_service import get_mongodb_service

    limit = int(request.GET.get('limit', 10))
    mongodb = get_mongodb_service()

    movies = Movie.objects.annotate(
        section_count=models.Count('sections')
    ).filter(
        section_count__lt=8
    ).order_by('id')[:limit]

    result = []
    for movie in movies:
        # Count embeddings in MongoDB
        embeddings_count = mongodb.get_embeddings_count(movie_id=movie.id)

        result.append({
            'id': movie.id,
            'title': movie.title,
            'year': movie.year,
            'tmdb_id': movie.tmdb_id,
            'sections_count': movie.section_count,
            'embeddings_count': embeddings_count
        })
    
    return JsonResponse({
        'count': len(result),
        'movies': result
    })


@require_http_methods(["GET"])
def get_movie_sections(request, movie_id):
    try:
        from services.mongodb_service import get_mongodb_service

        movie = Movie.objects.get(id=movie_id)
        sections = MovieSection.objects.filter(movie=movie)
        mongodb = get_mongodb_service()

        result = {}
        for section in sections:
            # Check MongoDB for embedding
            mongo_doc = mongodb.get_embedding(section.id)
            has_embedding = mongo_doc is not None

            result[section.section_type] = {
                'id': section.id,
                'section_type': section.section_type,
                'word_count': section.word_count,
                'has_embedding': has_embedding
            }

        return JsonResponse({
            'movie_id': movie.id,
            'movie_title': movie.title,
            'sections': result
        })
        
    except Movie.DoesNotExist:
        return JsonResponse({'error': 'Movie not found'}, status=404)


def get_director(movie_data):
    if 'credits' in movie_data and 'crew' in movie_data['credits']:
        for person in movie_data['credits']['crew']:
            if person['job'] == 'Director':
                return person['name']
    return 'Unknown Director'