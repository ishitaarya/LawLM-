from download_dataset import main as download_main
from clean_dataset import main as clean_main
from split_dataset import main as split_main
from dataset_stats import main as stats_main


if __name__ == "__main__":
    print("\n[1/4] Downloading dataset...")
    download_main()

    print("\n[2/4] Cleaning dataset...")
    clean_main()

    print("\n[3/4] Creating train/validation/test splits...")
    split_main()

    print("\n[4/4] Generating dataset statistics...")
    stats_main()

    print("\nLawLM Phase 2 dataset pipeline completed successfully.")
