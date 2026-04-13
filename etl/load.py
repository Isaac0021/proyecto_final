import polars as pl
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME")

PROCESSED_DIR     = os.path.join(BASE_DIR, "..", "data", "processed")
BATCH_SIZE        = 10_000

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
        logging.FileHandler(os.path.join(log_dir, "load.log"), mode="w"),
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


def insert_processed(client: MongoClient, df: pl.DataFrame):
    db         = client[DB_NAME]
    collection = db["processed"]

    log.info("Dropping existing 'processed' collection...")
    collection.drop()

    records = df.to_dicts()
    total   = len(records)
    log.info("Inserting %s documents into 'processed'...", f"{total:,}")

    for i in range(0, total, BATCH_SIZE):
        collection.insert_many(records[i : i + BATCH_SIZE], ordered=False)
        log.info("  Inserted batch %s / %s", f"{min(i + BATCH_SIZE, total):,}", f"{total:,}")

    # Indexes useful for the recommendation app
    collection.create_index([("userId",  ASCENDING)])
    collection.create_index([("movieId", ASCENDING)])
    collection.create_index([("avg_rating", ASCENDING)])
    collection.create_index([("rating_count", ASCENDING)])

    final_count = collection.count_documents({})
    log.info("'processed' collection ready — %s documents.", f"{final_count:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== LOAD STEP START ===")

    processed_path = os.path.join(PROCESSED_DIR, "processed.parquet")
    log.info("Reading processed parquet from: %s", processed_path)
    processed = pl.read_parquet(processed_path)
    log.info("processed shape: %s rows x %s cols", f"{processed.height:,}", processed.width)

    client = connect_mongo()
    insert_processed(client, processed)

    client.close()
    log.info("=== LOAD STEP COMPLETE ===")


if __name__ == "__main__":
    main()