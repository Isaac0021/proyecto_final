import polars as pl
from dotenv import load_dotenv
import os
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

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
        logging.FileHandler(os.path.join(log_dir, "transform.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cargar archvivos parquet creados por utilizar la libreria polars para crear
# dataframes.
# ---------------------------------------------------------------------------

def load_extracted() -> tuple:
    log.info("Loading extracted parquet files...")
    ratings = pl.read_parquet(os.path.join(PROCESSED_DIR, "ratings_raw.parquet"))
    movies  = pl.read_parquet(os.path.join(PROCESSED_DIR, "movies_raw.parquet"))
    tags    = pl.read_parquet(os.path.join(PROCESSED_DIR, "tags_raw.parquet"))
    log.info("Loaded — ratings: %s | movies: %s | tags: %s",
             f"{ratings.height:,}", f"{movies.height:,}", f"{tags.height:,}")
    return ratings, movies, tags


# ---------------------------------------------------------------------------
# Ratings películas.
# ---------------------------------------------------------------------------

def clean_ratings(ratings: pl.DataFrame) -> pl.DataFrame:
    log.info("Cleaning ratings...")
    before = ratings.height

    ratings = ratings.drop_nulls()

    ratings = ratings.unique(subset=["userId", "movieId"], keep="last")

    ratings = ratings.filter(pl.col("rating").is_between(0.5, 5.0))

    ratings = ratings.with_columns([
        pl.col("userId").cast(pl.Int64),
        pl.col("movieId").cast(pl.Int64),
        pl.col("rating").cast(pl.Float32),
        pl.col("timestamp").cast(pl.Int64),
    ])

    log.info("Ratings cleaned — %s → %s rows (dropped %s)", f"{before:,}", f"{ratings.height:,}", f"{before - ratings.height:,}")
    return ratings


def clean_movies(movies: pl.DataFrame) -> pl.DataFrame:
    log.info("Cleaning movies...")

    movies = movies.drop_nulls(subset=["movieId", "title"])

    movies = movies.with_columns(
        pl.col("title")
        .str.extract(r"\((\d{4})\)$", 1)
        .cast(pl.Int32, strict=False)
        .alias("year")
    )

    movies = movies.with_columns(
        pl.col("title").str.replace(r"\s*\(\d{4}\)$", "").str.strip_chars().alias("title")
    )

    log.info("Movies cleaned — %s rows", f"{movies.height:,}")
    return movies


def clean_tags(tags: pl.DataFrame) -> pl.DataFrame:
    log.info("Cleaning tags...")

    tags = tags.drop_nulls()
    tags = tags.with_columns(
        pl.col("tag").str.to_lowercase().str.strip_chars()
    )
    tags = tags.unique(subset=["userId", "movieId", "tag"])

    log.info("Tags cleaned — %s rows", f"{tags.height:,}")
    return tags


# ---------------------------------------------------------------------------
# Crear nuevo dataset, unificando los diferentes csv's que tenemos para
# poder accesar y predecir sobre una sola fuente.
# ---------------------------------------------------------------------------

def compute_movie_stats(ratings: pl.DataFrame) -> pl.DataFrame:
    log.info("Computing movie-level stats...")
    stats = ratings.group_by("movieId").agg([
        pl.col("rating").mean().alias("avg_rating"),
        pl.col("rating").count().alias("rating_count"),
        pl.col("rating").std().alias("rating_std"),
    ]).with_columns([
        pl.col("avg_rating").round(3),
        pl.col("rating_std").fill_null(0.0).round(3),
    ])
    log.info("Movie stats computed — %s movies", f"{stats.height:,}")
    return stats

def compute_user_stats(ratings: pl.DataFrame) -> pl.DataFrame:
    log.info("Computing user-level stats...")
    stats = ratings.group_by("userId").agg([
        pl.col("rating").count().alias("user_rating_count"),
        pl.col("rating").mean().alias("user_avg_rating"),
    ]).with_columns(
        pl.col("user_avg_rating").round(3)
    )
    log.info("User stats computed — %s users", f"{stats.height:,}")
    return stats

def aggregate_tags(tags: pl.DataFrame) -> pl.DataFrame:
    log.info("Aggregating tags per movie...")
    agg = tags.group_by("movieId").agg(
        pl.col("tag").alias("tags")
    )
    log.info("Tags aggregated — %s movies with tags", f"{agg.height:,}")
    return agg


# ---------------------------------------------------------------------------
# Construir nuevo dataset con cambios hechos.
# ---------------------------------------------------------------------------

def build_processed(
    ratings: pl.DataFrame,
    movies: pl.DataFrame,
    movie_stats: pl.DataFrame,
    user_stats: pl.DataFrame,
    tag_agg: pl.DataFrame,
) -> pl.DataFrame:
    log.info("Building processed dataset...")

    movies_enriched = movies.join(movie_stats, on="movieId", how="left")

    movies_enriched = movies_enriched.join(tag_agg, on="movieId", how="left")

    movies_enriched = movies_enriched.with_columns(
        pl.col("tags").fill_null([])
    )

    processed = (
        ratings
        .join(movies_enriched, on="movieId", how="left")
        .join(user_stats, on="userId", how="left")
    )

    log.info("processed dataset built — %s rows x %s cols", processed.height, processed.width)
    log.info("Columns: %s", processed.columns)
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== TRANSFORM STEP START ===")

    ratings, movies, tags = load_extracted()

    log.info("--- Cleaning ---")
    ratings = clean_ratings(ratings)
    movies  = clean_movies(movies)
    tags    = clean_tags(tags)

    log.info("--- Feature Engineering ---")
    movie_stats = compute_movie_stats(ratings)
    user_stats  = compute_user_stats(ratings)
    tag_agg     = aggregate_tags(tags)

    log.info("--- Building processed Dataset ---")
    processed = build_processed(ratings, movies, movie_stats, user_stats, tag_agg)

    # Save for load step
    out_path = os.path.join(PROCESSED_DIR, "processed.parquet")
    processed.write_parquet(out_path)
    log.info("processed dataset saved to: %s", out_path)

    log.info("=== TRANSFORM STEP COMPLETE ===")


if __name__ == "__main__":
    main()