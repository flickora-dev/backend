import openai
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class OpenRouterService:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENROUTER_API_KEY
        )
        self.model = "google/gemma-3-4b-it:free"
    
    def generate_movie_section(self, movie_data, section_type):
        try:
            prompt = self._create_section_prompt(movie_data, section_type)
            
            target_words = self._get_target_words(section_type)
            max_tokens = int(target_words * 1.5)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating section {section_type} for {movie_data.get('title', 'Unknown')}: {e}")
            return None
    
    def _get_target_words(self, section_type):
        """Return target word count for section type"""
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
        return targets.get(section_type, 500)
    
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
        
        genres_str = movie_data.get('genres', '') if isinstance(movie_data.get('genres'), str) else ', '.join([g.name for g in movie_data.get('genres', [])])
        
        target_words = self._get_target_words(section_type)
        
        prompt = f"""You are a movie enthusiast writing accessible, engaging analysis for everyday film lovers.

Movie: "{movie_data.get('title', '')}" ({movie_data.get('year', '')})
Director: {movie_data.get('director', '')}
Genres: {genres_str}
Plot: {movie_data.get('plot_summary', '')}

{instruction}

CRITICAL RULES:
- Write EXACTLY {target_words} words (not {target_words-50}, not {target_words+50})
- Use SIMPLE, EVERYDAY language - avoid complex vocabulary
- NO titles, NO headings, NO section labels, NO hashtags
- Start directly with the content
- Write in flowing paragraphs that anyone can understand
- Be specific with examples and details
- Write like talking to a friend about the movie

Begin writing the {target_words}-word analysis NOW:"""
        
        return prompt