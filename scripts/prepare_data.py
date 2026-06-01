"""
Chuyển dataset Tiki (raw .txt) sang CSV thống nhất cho pipeline ML.
Nguồn: https://github.com/quangvh23/tiki-dataset-sentiment-classification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = [
    "product_id",
    "reviewer",
    "rating",
    "title",
    "helpful_votes",
    "verified",
    "time_ago",
    "text",
]

RATING_TO_LABEL = {
    20: "tieu_cuc",
    40: "tieu_cuc",
    60: "trung_lap",
    80: "tich_cuc",
    100: "tich_cuc",
}


def rating_to_sentiment(rating: int | str) -> str:
    return RATING_TO_LABEL[int(rating)]


def load_raw_folder(raw_dir: Path, variant: str = "1a3") -> pd.DataFrame:
    """Đọc toàn bộ file .txt trong thư mục variant (1a3 = text gốc, 1a6 = đã token hóa)."""
    rows: list[dict] = []
    source_dir = raw_dir / variant

    for file_path in sorted(source_dir.glob("*.txt")):
        category = file_path.stem
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 8:
                    continue
                row = dict(zip(COLUMNS, parts[:8]))
                row["category"] = category
                row["variant"] = variant
                rows.append(row)

    df = pd.DataFrame(rows)
    df["rating"] = df["rating"].astype(int)
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    df["verified"] = df["verified"].map({"True": True, "False": False})
    df["label"] = df["rating"].map(rating_to_sentiment)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tiki review dataset")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/tiki-dataset-sentiment-classification"),
        help="Thư mục repo dataset đã clone",
    )
    parser.add_argument(
        "--variant",
        choices=["1a3", "1a6"],
        default="1a3",
        help="1a3: văn bản gốc | 1a6: văn bản đã tách token bằng dấu _",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tiki_reviews.csv"),
        help="File CSV đầu ra",
    )
    args = parser.parse_args()

    df = load_raw_folder(args.raw_dir, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df):,} rows -> {args.output}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nRating distribution:")
    print(df["rating"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
