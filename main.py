# from openai import OpenAI


# completion = client.chat.completions.create(

#   model="deepseek/deepseek-chat-v3.1:free",
#   messages=[
#     {
#       "role": "user",
#       "content": """You are an expert film critic writing a comprehensive 3000-word analysis for "The Shawshank Redemption"

# STRUCTURE: 6 sections × 500 words each = 3000 words total

# ## BASIC INFORMATION (500 words)
# Production details, budget, box office, initial casting announcements, crew, awards, release strategy.

# ## CAST & PERFORMANCES (500 words) - ACTOR FOCUS
# Detailed analysis of each main actor's performance:
# - Acting techniques and preparation methods
# - Casting decisions and audition process
# - On-screen chemistry and relationships
# - Memorable scenes showcasing talent
# - How performances serve the story
# - Critical reception of individual performances
# - Career context for each actor

# ## CHARACTER ANALYSIS (500 words) - CHARACTER FOCUS
# Deep dive into character psychology and development:
# - Character motivations and internal conflicts
# - Character arcs and transformation
# - Relationships between characters
# - Symbolic and archetypal functions
# - How characters embody themes
# - Character writing and dialogue
# - Psychological complexity and depth

# ## THEMATIC & ARTISTIC ELEMENTS (500 words)
# Central themes, directorial vision, cinematography, production design, score, editing, visual symbolism.

# ## CRITICAL RECEPTION & REVIEWS (500 words) - REVIEWS FOCUS
# Comprehensive review analysis:
# - Major critics' opinions (quote specific reviews)
# - Professional publication reviews
# - Audience reception and ratings
# - Festival response and awards recognition
# - Evolution of critical opinion over time
# - Controversial aspects or debates
# - Modern reassessment

# ## LEGACY & CULTURAL IMPACT (500 words)
# Long-term influence, cultural significance, place in cinema history, ongoing relevance.

# MOVIE DATA:
# Title: {movie_title}
# Year: {year}
# Director: {director}
# Main Cast: {main_cast}
# Genre: {genre}

# Write all 6 sections (3000 words total), focusing heavily on actors, characters, and reviews as requested."""
#     }
#   ]
# )

# print(completion.choices[0].message.content)
import requests

url = "https://api.themoviedb.org/3/account/22296958/rated/movies?language=en-US&page=1&sort_by=created_at.asc"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1ZDBjOTcyMWUwYmVlODNkOWE2OTM5YjhiNzkxOWI1NiIsIm5iZiI6MTc1NzQxMjIwMy4yNDEsInN1YiI6IjY4YmZmYjZiODU3YjkxYTRjMThjZmI2ZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Pbku44mN0A7CigrO0X8870mYdpLoU5aVgtIv1hMkHCc",
}

response = requests.get(url, headers=headers)

print(response.text)
