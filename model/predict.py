from surprise import SVD
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cargar modelo
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    model_path = os.path.join(ARTIFACTS_DIR, "svd_model.joblib")
    train_path = os.path.join(ARTIFACTS_DIR, "trainset.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run model/train.py first.")

    model    = joblib.load(model_path)
    trainset = joblib.load(train_path)
    log.info("Model and trainset loaded.")
    return model, trainset


# ---------------------------------------------------------------------------
# Extraer datos de cada película y los mete en un diccionario.
# ---------------------------------------------------------------------------

def fetch_all_movies(client: MongoClient) -> dict:
    db     = client[DB_NAME]
    cursor = db["movies"].find({}, {"movieId": 1, "title": 1, "genres": 1, "_id": 0})
    return {doc["movieId"]: doc for doc in cursor}


# ---------------------------------------------------------------------------
# El usuario califica mínimo 3 películas, el modelo toma los factores
# de las películas que le gustaron (≥3.5 estrellas) para
# construir un "vector de gustos", y luego encuentra las películas más
# similares a ese vector usando similitud coseno.
# ---------------------------------------------------------------------------

def recommend(user_ratings: list[dict], top_n: int = 10) -> list[dict]:
    import numpy as np

    model, trainset = load_model()

    client     = MongoClient(MONGO_URI)
    all_movies = fetch_all_movies(client)
    client.close()

    rated_ids     = {r["movieId"] for r in user_ratings}
    high_rated    = [r for r in user_ratings if r["rating"] >= 3.5]

    if not high_rated:
        high_rated = user_ratings

    liked_factors = []
    for r in high_rated:
        try:
            iid = trainset.to_inner_iid(r["movieId"])
            liked_factors.append(model.qi[iid] * r["rating"])
        except ValueError:
            continue

    if not liked_factors:
        predictions = []
        for movie_id, movie_info in all_movies.items():
            if movie_id in rated_ids:
                continue
            pred = model.predict(uid="new_user", iid=movie_id)
            predictions.append({
                "movieId":          movie_id,
                "title":            movie_info.get("title", "Unknown"),
                "genres":           movie_info.get("genres", []),
                "predicted_rating": round(pred.est, 2),
            })
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return predictions[:top_n]

    taste_vector = np.mean(liked_factors, axis=0)

    predictions = []
    for movie_id, movie_info in all_movies.items():
        if movie_id in rated_ids:
            continue
        try:
            iid    = trainset.to_inner_iid(movie_id)
            factor = model.qi[iid]
            # Cosine similarity
            sim = float(np.dot(taste_vector, factor) / (
                np.linalg.norm(taste_vector) * np.linalg.norm(factor) + 1e-9
            ))
            predictions.append({
                "movieId":          movie_id,
                "title":            movie_info.get("title", "Unknown"),
                "genres":           movie_info.get("genres", []),
                "predicted_rating": round(sim * 5, 2),  # scale to 0-5 range
            })
        except ValueError:
            continue

    predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
    log.info("Returning top %s recommendations.", top_n)
    return predictions[:top_n]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()