import polars as pl
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME")

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
        logging.FileHandler(os.path.join(log_dir, "extract.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connect_mongo() -> MongoClient:
    log.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    client.admin.command("ping")
    log.info("Connected — database: %s", DB_NAME)
    return client


def collection_to_df(client: MongoClient, collection_name: str, projection: dict) -> pl.DataFrame:
    log.info("Extracting collection: '%s'...", collection_name)
    db      = client[DB_NAME]
    cursor  = db[collection_name].find({}, projection)
    records = list(cursor)

    # Remove MongoDB's _id field
    for r in records:
        r.pop("_id", None)

    df = pl.DataFrame(records)
    log.info("'%s' extracted — %s rows x %s cols", collection_name, df.height, df.width)
    return df


# ---------------------------------------------------------------------------
# Extract functions
# ---------------------------------------------------------------------------

def extract_ratings(client: MongoClient) -> pl.DataFrame:
    return collection_to_df(
        client, "ratings",
        projection={"userId": 1, "movieId": 1, "rating": 1, "timestamp": 1}
    )


def extract_movies(client: MongoClient) -> pl.DataFrame:
    return collection_to_df(
        client, "movies",
        projection={"movieId": 1, "title": 1, "genres": 1}
    )


def extract_tags(client: MongoClient) -> pl.DataFrame:
    return collection_to_df(
        client, "tags",
        projection={"userId": 1, "movieId": 1, "tag": 1}
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== EXTRACT STEP START ===")
    client = connect_mongo()

    ratings = extract_ratings(client)
    movies  = extract_movies(client)
    tags    = extract_tags(client)

    # Save to processed folder for transform step to pick up
    processed_dir = os.path.join(BASE_DIR, "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    ratings.write_parquet(os.path.join(processed_dir, "ratings_raw.parquet"))
    movies.write_parquet(os.path.join(processed_dir, "movies_raw.parquet"))
    tags.write_parquet(os.path.join(processed_dir, "tags_raw.parquet"))

    log.info("Saved extracted data to data/processed/ as parquet files.")
    log.info("=== EXTRACT STEP COMPLETE ===")

    client.close()


if __name__ == "__main__":
    main()