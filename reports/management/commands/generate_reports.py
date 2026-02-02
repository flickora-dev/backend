from django.core.management.base import BaseCommand
from django.conf import settings
from movies.models import Movie
from reports.models import MovieSection
from services.rag_service import RAGService
from services.mongodb_service import MongoDBVectorService
import requests
import time
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate AI reports for movies with automatic embedding generation'

    def add_arguments(self, parser):
        parser.add_argument('--movie-id', type=int, help='Generate report for specific movie')
        parser.add_argument('--all', action='store_true', help='Generate reports for all movies without reports')
        parser.add_argument('--limit', type=int, default=5, help='Limit number of movies to process')
        parser.add_argument('--skip-embeddings', action='store_true', help='Skip embedding generation')

    def handle(self, *args, **options):
        rag = RAGService()
        mongodb = MongoDBVectorService()

        # Ollama configuration
        ollama_url = settings.OLLAMA_BASE_URL.replace('/v1', '')
        ollama_model = settings.OLLAMA_MODEL

        if options['movie_id']:
            movies = Movie.objects.filter(id=options['movie_id'])
        elif options['all']:
            movies = Movie.objects.filter(sections__isnull=True).distinct()[:options['limit']]
        else:
            movies = Movie.objects.filter(sections__isnull=True).distinct()[:options['limit']]

        if not movies:
            self.stdout.write(self.style.WARNING('No movies to process'))
            return

        section_types = [
            'production',
            'plot_structure',
            'cast_crew',
            'characters',
            'visual_technical',
            'themes',
            'reception',
            'legacy'
        ]

        total_generated = 0

        for movie in movies:
            self.stdout.write(f"\nProcessing: {movie.title} ({movie.year})")

            movie_data = {
                'title': movie.title,
                'year': movie.year,
                'director': movie.director,
                'genres': ', '.join([g.name for g in movie.genres.all()]),
                'plot_summary': movie.plot_summary
            }

            for section_type in section_types:
                if MovieSection.objects.filter(movie=movie, section_type=section_type).exists():
                    self.stdout.write(f"  - {section_type}: already exists")
                    continue

                try:
                    self.stdout.write(f"  - Generating {section_type}...")

                    content = self._generate_section(
                        ollama_url, ollama_model, movie_data, section_type
                    )

                    if content:
                        # Create section first
                        section = MovieSection.objects.create(
                            movie=movie,
                            section_type=section_type,
                            content=content
                        )

                        # Generate and store embedding in MongoDB
                        embedding_stored = False
                        if not options['skip_embeddings']:
                            self.stdout.write(f"    - Generating embedding...")
                            try:
                                embedding = rag.generate_embedding(content)
                                if embedding is not None:
                                    success = mongodb.store_embedding(
                                        section_id=section.id,
                                        movie_id=movie.id,
                                        section_type=section_type,
                                        embedding=embedding.tolist(),
                                        metadata={
                                            'word_count': section.word_count,
                                            'movie_title': movie.title
                                        }
                                    )
                                    embedding_stored = success
                            except Exception as e:
                                self.stdout.write(self.style.WARNING(f"    ! Embedding generation failed: {e}"))

                        total_generated += 1
                        emb_status = 'yes' if embedding_stored else 'no'
                        self.stdout.write(self.style.SUCCESS(
                            f"    ✓ Generated ({len(content.split())} words, embedding: {emb_status})"
                        ))
                    else:
                        self.stdout.write(self.style.ERROR(f"    ✗ Failed to generate"))

                    time.sleep(1)

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"    ✗ Error: {e}"))

        self.stdout.write(self.style.SUCCESS(f"\nTotal sections generated: {total_generated}"))

    def _generate_section(self, ollama_url, model, movie_data, section_type):
        """Generate a movie section using Ollama"""
        try:
            prompt = self._create_section_prompt(movie_data, section_type)
            target_words = self._get_target_words(section_type)

            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": int(target_words * 3)  # more tokens to reach word count
                    }
                },
                timeout=180
            )
            response.raise_for_status()
            result = response.json()

            return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Error generating section {section_type} for {movie_data.get('title', 'Unknown')}: {e}")
            return None

    def _get_target_words(self, section_type):
        """Return target word count for section type (inflated to compensate for model output)"""
        # Model typically produces 70-80% of requested words, so we inflate targets
        targets = {
            'production': 550,
            'plot_structure': 800,
            'cast_crew': 550,
            'characters': 700,
            'visual_technical': 700,
            'themes': 700,
            'reception': 550,
            'legacy': 550,
        }
        return targets.get(section_type, 600)

    def _create_section_prompt(self, movie_data, section_type):
        section_instructions = {
            'production': """Write EXACTLY 400 words analyzing production and release.

            COVER:
            - Production: budget, filming locations, schedule, challenges
            - Box Office: opening weekend, total gross, profitability
            - Awards: major nominations, wins (Oscars, Golden Globes, etc.)
            - Release Strategy: date, marketing, distribution
            - Behind-the-scenes stories and interesting production facts

            Use SIMPLE, CONVERSATIONAL language. Write like explaining to a friend.""",

            'plot_structure': """Write EXACTLY 600 words providing detailed plot breakdown.

            STRUCTURE YOUR ANALYSIS:
            - Act 1 Setup (150 words): Opening scenes, character introductions, inciting incident
            - Act 2 Confrontation (250 words): Rising action, complications, major plot points, conflicts
            - Act 3 Resolution (150 words): Climax, falling action, resolution, ending
            - Key Turning Points (50 words): Major twists and their impact

            INCLUDE:
            - Character motivations driving the plot
            - How subplots interconnect with main story
            - Pacing and narrative structure
            - Significance of key scenes

            THIS IS THE MOST CRITICAL SECTION - be thorough but clear.
            Use simple language that anyone can understand.""",

            'cast_crew': """Write EXACTLY 400 words about the filmmaking team.

            COVER:
            - Director: background, style, previous works, approach to this film
            - Main Actors: 3-5 leads with their preparation and performance
            - Chemistry: how actors work together
            - Casting: interesting casting decisions and audition stories
            - Key Crew: cinematographer, composer, production designer (mention if notable)

            Use conversational language. Focus on people and their contributions.""",

            'characters': """Write EXACTLY 500 words analyzing characters deeply.

            ANALYZE 3-5 MAIN CHARACTERS:
            For each character discuss:
            - Core motivations and desires
            - Internal conflicts and struggles
            - Character arc (how they change)
            - Relationships with other characters
            - What they represent symbolically

            INCLUDE:
            - Archetypes used
            - Psychological depth and complexity
            - How characters drive the story

            Write clearly and accessibly.""",

            'visual_technical': """Write EXACTLY 500 words analyzing technical craftsmanship.

            STRUCTURE:
            - Cinematography (150 words): camera work, shot composition, lighting style, visual choices
            - Production Design (100 words): sets, costumes, props, color palette
            - Editing (100 words): pacing, transitions, montage techniques, rhythm
            - Sound & Music (100 words): score, soundtrack, sound design, use of silence
            - Visual Effects (50 words): CGI, practical effects, special techniques

            Be specific about techniques and their storytelling impact.
            Use clear explanations for technical terms.""",

            'themes': """Write EXACTLY 500 words analyzing themes and symbolism.

            IDENTIFY 2-4 CENTRAL THEMES:
            For each theme discuss:
            - How it's presented in the story
            - Visual and narrative symbolism
            - Character embodiment of theme
            - Philosophical questions raised

            INCLUDE:
            - Director's vision and message
            - Social/cultural commentary
            - Deeper meanings and subtext
            - How themes interconnect

            Write thoughtfully but accessibly.""",

            'reception': """Write EXACTLY 400 words analyzing critical reception.

            COVER:
            - Ratings: IMDb, Rotten Tomatoes, Metacritic scores
            - Critical Consensus: what most critics agreed on
            - Positive Reviews: praised aspects with examples
            - Criticisms: what was criticized
            - Audience vs Critics: differences in reception
            - Evolution: how opinions changed over time
            - Controversies: debates or polarizing elements

            Use clear, objective language.""",

            'legacy': """Write EXACTLY 400 words analyzing cultural impact.

            COVER:
            - Influence on Cinema: films it inspired, techniques it popularized
            - Cultural Significance: impact on pop culture, memorable elements
            - Fan Community: cult following, fan theories, ongoing discussions
            - Historical Place: position in film history
            - Modern Relevance: why it still matters today
            - Lasting Innovations: what it contributed to filmmaking

            Write engagingly about the film's enduring importance.""",
        }

        instruction = section_instructions.get(section_type, "Write 500 words of analysis in simple, conversational language.")

        genres_str = movie_data.get('genres', '')
        target_words = self._get_target_words(section_type)

        prompt = f"""You are a movie enthusiast writing accessible, engaging analysis for everyday film lovers.

Movie: "{movie_data.get('title', '')}" ({movie_data.get('year', '')})
Director: {movie_data.get('director', '')}
Genres: {genres_str}
Plot: {movie_data.get('plot_summary', '')}

{instruction}

ABSOLUTE REQUIREMENTS:
1. You MUST write AT LEAST {target_words} words - this is mandatory
2. Count your words as you write - do NOT stop before reaching {target_words}
3. If you finish your main points early, add more details, examples, and analysis
4. Use SIMPLE, EVERYDAY language
5. NO titles, NO headings, NO section labels, NO hashtags
6. Start directly with the content
7. Write in flowing paragraphs

IMPORTANT: Your response will be rejected if it has fewer than {target_words} words. Keep writing until you reach the word count.

Begin writing the {target_words}-word analysis NOW:"""

        return prompt
