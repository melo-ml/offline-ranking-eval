import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.config import RAW_DATA_DIR, ensure_project_dirs

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens_1m(raw_dir: Path, force: bool = False) -> Path:
    dataset_dir = raw_dir / "ml-1m"
    ratings_path = dataset_dir / "ratings.dat"

    if ratings_path.exists() and not force:
        print(f"Dataset already exists at {ratings_path}")
        return dataset_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-1m.zip"

    print(f"Downloading MovieLens 1M from {MOVIELENS_1M_URL}")
    urlretrieve(MOVIELENS_1M_URL, zip_path)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    if zip_path.exists():
        zip_path.unlink()

    if not ratings_path.exists():
        raise FileNotFoundError(f"Expected file not found after extraction: {ratings_path}")

    print(f"MovieLens 1M ready at: {dataset_dir}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MovieLens 1M dataset.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    return parser.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    download_movielens_1m(args.raw_dir, force=args.force)


if __name__ == "__main__":
    main()
