"""Run the complete Phase 2 corpus pipeline."""

from src.data.download_dataset import main as download_main
from src.data.clean_dataset import main as clean_main
from src.data.split_dataset import main as split_main
from src.data.dataset_stats import main as stats_main


if __name__ == "__main__":
    print("\n[1/4] Collecting legal provisions...")
    download_main()

    print("\n[2/4] Cleaning and deduplicating...")
    clean_main()

    print("\n[3/4] Creating train/validation/test splits...")
    split_main()

    print("\n[4/4] Generating corpus statistics...")
    stats_main()

    print("\nLawSuit LLM Phase 2 pipeline completed.")
