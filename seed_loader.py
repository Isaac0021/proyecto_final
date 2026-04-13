import polars as pl
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
SAMPLE_USERS = 5_000
MIN_MOVIE_RATINGS = 100
RANDOM_SEED = 42
BATCH_SIZE = 10_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_dir = os.path.join(BASE_DIR, "orchestration", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "seed_loader.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def connect_mongo() -> MongoClient:
    log.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    client.admin.command("ping")
    log.info("Connected to database: %s", DB_NAME)
    return client


def load_ratings() -> pl.DataFrame:
    path = os.path.join(DATA_DIR, "ratings.csv")
    log.info("Reading ratings.csv with Polars...")
    df = pl.read_csv(path, schema_overrides={
        "userId": pl.Int64,
        "movieId": pl.Int64,
        "rating": pl.Float32,
        "timestamp": pl.Int64,
    })
    log.info("Full ratings shape: %s rows x %s cols", df.height, df.width)
    return df


def sample_ratings(ratings: pl.DataFrame) -> tuple:
    log.info("Sampling %s random users...", f"{SAMPLE_USERS:,}")
    all_users = ratings["userId"].unique()
    sampled_users = all_users.sample(n=min(SAMPLE_USERS, len(all_users)), seed=RANDOM_SEED)
    ratings = ratings.filter(pl.col("userId").is_in(sampled_users))
    log.info("Ratings after user filter: %s", f"{ratings.height:,}")

    log.info("Filtering movies with fewer than %s ratings...", MIN_MOVIE_RATINGS)
    movie_counts = (
        ratings
        .group_by("movieId")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= MIN_MOVIE_RATINGS)
    )
    ratings = ratings.filter(pl.col("movieId").is_in(movie_counts["movieId"]))
    log.info("Ratings after movie popularity filter: %s", f"{ratings.height:,}")

    sampled_user_ids = set(ratings["userId"].to_list())
    sampled_movie_ids = set(ratings["movieId"].to_list())

    log.info(
        "Final sample — users: %s | movies: %s | ratings: %s",
        f"{len(sampled_user_ids):,}",
        f"{len(sampled_movie_ids):,}",
        f"{ratings.height:,}",
    )
    return ratings, sampled_user_ids, sampled_movie_ids


def load_movies(sampled_movie_ids: set) -> pl.DataFrame:
    log.info("Reading movies.csv...")
    df = pl.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    df = df.filter(pl.col("movieId").is_in(sampled_movie_ids))
    log.info("Movies loaded: %s", f"{df.height:,}")
    return df


def load_tags(sampled_user_ids: set, sampled_movie_ids: set) -> pl.DataFrame:
    log.info("Reading tags.csv...")
    df = pl.read_csv(
        os.path.join(DATA_DIR, "tags.csv"),
        schema_overrides={"userId": pl.Int64, "movieId": pl.Int64},
    )
    df = df.filter(
        pl.col("userId").is_in(sampled_user_ids) &
        pl.col("movieId").is_in(sampled_movie_ids)
    )
    log.info("Tags loaded: %s", f"{df.height:,}")
    return df


def load_links(sampled_movie_ids: set) -> pl.DataFrame:
    log.info("Reading links.csv...")
    df = pl.read_csv(
        os.path.join(DATA_DIR, "links.csv"),
        schema_overrides={"movieId": pl.Int64},
    )
    df = df.filter(pl.col("movieId").is_in(sampled_movie_ids))
    df = df.with_columns([
        pl.col("imdbId").fill_null(0),
        pl.col("tmdbId").fill_null(0),
    ])
    log.info("Links loaded: %s", f"{df.height:,}")
    return df


def insert_collection(
        client: MongoClient,
        collection_name: str,
        df: pl.DataFrame,
        indexes: list = None,
        unique_index: list = None,
):
    db = client[DB_NAME]
    collection = db[collection_name]
    collection.drop()
    log.info("Inserting %s docs into '%s'...", f"{df.height:,}", collection_name)

    if collection_name == "movies":
        records = [
            {
                **row,
                "genres": row["genres"].split("|") if row["genres"] != "(no genres listed)" else []
            }
            for row in df.to_dicts()
        ]
    else:
        records = df.to_dicts()

    for i in range(0, len(records), BATCH_SIZE):
        collection.insert_many(records[i: i + BATCH_SIZE], ordered=False)

    if indexes:
        for idx in indexes:
            collection.create_index([(idx, ASCENDING)])
    if unique_index:
        collection.create_index(
            [(field, ASCENDING) for field in unique_index],
            unique=True,
        )

    log.info("'%s' ready — %s documents.", collection_name, f"{collection.count_documents({}):,}")


def print_summary(client: MongoClient):
    db = client[DB_NAME]
    log.info("=" * 50)
    log.info("SEED COMPLETE — collection counts:")
    for name in ["movies", "ratings", "tags", "links"]:
        count = db[name].count_documents({})
        log.info("  %-10s %s documents", name, f"{count:,}")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Starting MovieLens seed loader...")

    client = connect_mongo()

    log.info("--- STEP 1: Load & sample ratings ---")
    ratings_full = load_ratings()
    ratings, user_ids, movie_ids = sample_ratings(ratings_full)

    log.info("--- STEP 2: Load supporting files ---")
    movies = load_movies(movie_ids)
    tags = load_tags(user_ids, movie_ids)
    links = load_links(movie_ids)

    log.info("--- STEP 3: Insert into MongoDB ---")
    insert_collection(client, "movies", movies,
                      indexes=["movieId", "genres"])
    insert_collection(client, "ratings", ratings,
                      indexes=["userId", "movieId"],
                      unique_index=["userId", "movieId"])
    insert_collection(client, "tags", tags,
                      indexes=["userId", "movieId"])
    insert_collection(client, "links", links,
                      indexes=["movieId"])

    log.info("--- STEP 4: Summary ---")
    print_summary(client)

    client.close()
    log.info("Seed loader finished successfully.")


if __name__ == "__main__":
    main()