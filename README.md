# Flickora — Backend

Backend REST API platformy Flickora do odkrywania filmów z AI. Zbudowany na Django 5.1 z obsługą wektorowego wyszukiwania semantycznego, czatu AI i integracji z TMDB.

## Stack

- **Django 5.1** + Django REST Framework
- **PostgreSQL** + pgvector
- **MongoDB Atlas** — embeddingi wektorowe
- **PyTorch + sentence-transformers** — model `all-MiniLM-L6-v2`
- **OpenRouter** — LLM do czatu (AI)
- **TMDB API** — metadane filmów
- **Gunicorn** — serwer WSGI
- JWT (djangorestframework-simplejwt)

## Uruchomienie lokalne

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Utwórz plik `.env`:
```env
SECRET_KEY=twoj-sekretny-klucz
DEBUG=True
DATABASE_URL=postgresql://user:haslo@localhost:5432/flickora
TMDB_API_KEY=twoj-klucz-tmdb
OPENROUTER_API_KEY=twoj-klucz-openrouter
MONGODB_URL=twoj-url-mongodb-atlas
MONGODB_DATABASE=flickora
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

```bash
python manage.py migrate
python manage.py runserver
```

API: `http://localhost:8000/api/`
Swagger: `http://localhost:8000/swagger/`

## Docker

```bash
docker build -t flickora-backend .
docker run -d -p 8000:8000 --env-file .env flickora-backend \
  bash -c "python manage.py migrate --noinput && gunicorn flickora.wsgi:application --bind 0.0.0.0:8000 --workers 2"
```

## Zmienne środowiskowe

| Zmienna | Wymagana | Opis |
|---|---|---|
| `SECRET_KEY` | Tak | Sekretny klucz Django |
| `DEBUG` | Nie | `True` dla devu, `False` dla produkcji |
| `DATABASE_URL` | Nie | URL PostgreSQL. Jeśli brak — używa SQLite |
| `TMDB_API_KEY` | Tak | Klucz API The Movie Database |
| `OPENROUTER_API_KEY` | Tak | Klucz API OpenRouter (LLM) |
| `MONGODB_URL` | Tak | Connection string MongoDB Atlas |
| `MONGODB_DATABASE` | Nie | Nazwa bazy danych (domyślnie: `flickora`) |
| `ALLOWED_HOSTS` | Nie | Hosty oddzielone przecinkiem |
| `CORS_ALLOWED_ORIGINS` | Nie | Originy oddzielone przecinkiem |
| `N8N_WEBHOOK_URL` | Nie | Webhook n8n do automatyzacji |

## Endpointy API

| Metoda | Endpoint | Opis |
|---|---|---|
| `GET` | `/api/movies/` | Lista filmów (wyszukiwanie, filtry, paginacja) |
| `GET` | `/api/movies/{id}/` | Szczegóły filmu z sekcjami analizy |
| `GET` | `/api/genres/` | Lista gatunków z liczbą filmów |
| `GET` | `/api/movies/{id}/similar/` | Podobne filmy (wyszukiwanie wektorowe) |
| `POST` | `/api/chat/send/` | Wyślij wiadomość czatu (odpowiedź AI) |
| `GET` | `/api/chat/conversations/` | Lista rozmów użytkownika |
| `POST` | `/api/auth/login/` | Logowanie (zwraca JWT) |
| `POST` | `/api/auth/register/` | Rejestracja nowego użytkownika |
| `POST` | `/api/auth/token/refresh/` | Odświeżenie tokenu JWT |
| `GET/PUT` | `/api/auth/profile/` | Pobierz / zaktualizuj profil |
| `GET` | `/api/health/` | Health check |

Pełna interaktywna dokumentacja: `/swagger/`

## Wdrożenie (Railway)

Konfiguracja w `railway.toml` — wykonuje migracje i uruchamia gunicorn automatycznie przy każdym deploymencie.

## Licencja

MIT
