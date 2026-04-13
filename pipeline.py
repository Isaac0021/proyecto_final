import subprocess
import sys
import os
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        logging.FileHandler(os.path.join(log_dir, "pipeline.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_step(step_number: int, name: str, script: str):
    log.info("")
    log.info("=" * 50)
    log.info("STEP %s — %s", step_number, name)
    log.info("Script: %s", script)
    log.info("Started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 50)

    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script)],
        capture_output=True,
        text=True,
    )

    # Forward the step's own logs to pipeline log
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log.info("  %s", line)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log.warning("  %s", line)

    if result.returncode != 0:
        log.error("STEP %s FAILED — %s", step_number, name)
        log.error("Exit code: %s", result.returncode)
        sys.exit(1)

    log.info("STEP %s COMPLETE — %s", step_number, name)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main():
    start = datetime.now()
    log.info("")
    log.info("*" * 50)
    log.info("  MOVIELENS PIPELINE STARTING")
    log.info("  %s", start.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("*" * 50)

    run_step(1, "Extract",   "etl/extract.py")
    run_step(2, "Transform", "etl/transform.py")
    run_step(3, "Load",      "etl/load.py")

    end      = datetime.now()
    elapsed  = end - start

    log.info("")
    log.info("*" * 50)
    log.info("  PIPELINE FINISHED SUCCESSFULLY")
    log.info("  Completed at : %s", end.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  Total time   : %s", str(elapsed).split(".")[0])
    log.info("*" * 50)


if __name__ == "__main__":
    main()