from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_dir = os.path.join(BASE_DIR, "..", "orchestration", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "train.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cargar datos de base en Mongo
# ---------------------------------------------------------------------------

def load_ratings_from_mongo() -> list:
    log.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    client.admin.command("ping")
    log.info("Connected — loading ratings from 'processed' collection...")

    db      = client[DB_NAME]
    cursor  = db["processed"].find({}, {"userId": 1, "movieId": 1, "rating": 1, "_id": 0})
    records = list(cursor)
    client.close()

    log.info("Loaded %s records from MongoDB.", f"{len(records):,}")
    return records


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(records: list):
    log.info("Building Surprise dataset...")
    reader = Reader(rating_scale=(0.5, 5.0))

    data = Dataset.load_from_df(
        __import__("pandas").DataFrame(records)[["userId", "movieId", "rating"]],
        reader,
    )

    log.info("Splitting into train/test (80/20)...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    log.info("Training SVD model...")
    model = SVD(n_factors=1000, n_epochs=50, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)
    log.info("Training complete.")

    # Evaluate
    log.info("Evaluating on test set...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)

    log.info("RMSE : %.4f", rmse)
    log.info("MAE  : %.4f", mae)
    log.info("Interpretation: on average predictions are off by %.2f stars", rmse)

    return model, trainset, {"rmse": round(rmse, 4), "mae": round(mae, 4)}


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(model, trainset, metrics: dict):
    model_path   = os.path.join(ARTIFACTS_DIR, "svd_model.joblib")
    train_path   = os.path.join(ARTIFACTS_DIR, "trainset.joblib")
    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")

    joblib.dump(model,    model_path)
    joblib.dump(trainset, train_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model saved   → %s", model_path)
    log.info("Trainset saved → %s", train_path)
    log.info("Metrics saved  → %s", metrics_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== TRAIN STEP START ===")

    records              = load_ratings_from_mongo()
    model, trainset, metrics = train(records)
    save_artifacts(model, trainset, metrics)

    log.info("=== TRAIN STEP COMPLETE ===")
    log.info("Final metrics: RMSE=%.4f | MAE=%.4f", metrics["rmse"], metrics["mae"])


if __name__ == "__main__":
    main()