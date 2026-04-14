from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import sys
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

sys.path.append(os.path.join(BASE_DIR, ".."))
from model.predict import recommend

MONGO_URI    = os.getenv("MONGO_URI")
DB_NAME      = os.getenv("DB_NAME")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# MongoDB datos URI y orden de películas, luego se añade un api
# que muestra los posters de las películas.
# ---------------------------------------------------------------------------

def get_db():
    client = MongoClient(MONGO_URI)
    return client, client[DB_NAME]


def get_poster(tmdb_id: int) -> str:
    if not tmdb_id:
        return None
    try:
        url  = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
        res  = requests.get(url, timeout=3).json()
        path = res.get("poster_path")
        return f"https://image.tmdb.org/t/p/w300{path}" if path else None
    except:
        return None


def get_popular_movies(limit: int = 50) -> list:
    client, db = get_db()
    pipeline = [
        {"$group": {
            "_id": "$movieId",
            "avg_rating":   {"$avg": "$rating"},
            "rating_count": {"$sum": 1},
        }},
        {"$sort": {"rating_count": -1}},
        {"$limit": limit},
    ]
    stats      = {doc["_id"]: doc for doc in db["processed"].aggregate(pipeline)}
    movie_ids  = list(stats.keys())
    movies_raw = db["movies"].find({"movieId": {"$in": movie_ids}}, {"_id": 0})
    links      = {doc["movieId"]: doc.get("tmdbId") for doc in db["links"].find(
                    {"movieId": {"$in": movie_ids}}, {"movieId": 1, "tmdbId": 1, "_id": 0})}

    movies = []
    for m in movies_raw:
        mid = m["movieId"]
        movies.append({
            "movieId":      mid,
            "title":        m.get("title", "Unknown"),
            "genres":       m.get("genres", []),
            "avg_rating":   round(stats[mid]["avg_rating"], 2),
            "rating_count": stats[mid]["rating_count"],
            "poster":       get_poster(links.get(mid)),
        })

    movies.sort(key=lambda x: x["rating_count"], reverse=True)
    client.close()
    return movies


# ---------------------------------------------------------------------------
# Ruta para correr el app y api de posters de pelicula
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    movies = get_popular_movies(limit=50)
    return render_template("index.html", movies=movies)


@app.route("/recommend", methods=["POST"])
def get_recommendations():
    data         = request.get_json()
    user_ratings = data.get("ratings", [])

    if len(user_ratings) < 3:
        return jsonify({"error": "Please rate at least 3 movies."}), 400

    try:
        client, db = get_db()
        links      = {doc["movieId"]: doc.get("tmdbId") for doc in db["links"].find(
                        {}, {"movieId": 1, "tmdbId": 1, "_id": 0})}
        client.close()

        results = recommend(user_ratings, top_n=10)

        for r in results:
            tmdb_id  = links.get(r["movieId"])
            r["poster"] = get_poster(tmdb_id)

        return jsonify({"recommendations": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)