from pathlib import Path
import csv
import logging

from .ml.preprocess import validate_satellite_images
from .ml.predict import predict_flood

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "ml"
CSV_PATH = DATASET_DIR / "pairs.csv"


def load_image_pairs():

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    pairs = []

    with open(CSV_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        required = {"location", "before_image", "after_image"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns: {required}"
            )

        for row in reader:
            before = DATASET_DIR / row["before_image"].strip()
            after = DATASET_DIR / row["after_image"].strip()

            pairs.append({
                "location": row["location"].strip(),
                "before": before,
                "after": after
            })

    return pairs


def run_flood_detection():

    pairs = load_image_pairs()

    if not pairs:
        raise ValueError("No satellite image pairs found.")

    results = []

    for pair in pairs:

        try:
            # Validate before and after satellite images.
            validate_satellite_images(
                pair["before"],
                pair["after"]
            )

            # Run trained ML model.
            prediction = predict_flood(
                str(pair["before"]),
                str(pair["after"])
            )

            results.append({
                "location": pair["location"],
                "flooded": bool(prediction["flooded"]),
                "flood_probability": float(
                    prediction["flood_probability"]
                ),
                "status": (
                    "Flood detected"
                    if prediction["flooded"]
                    else "No flood detected"
                )
            })

        except Exception as error:
            logger.exception(
                "Prediction failed for %s",
                pair["location"]
            )

            results.append({
                "location": pair["location"],
                "error": str(error),
                "status": "Processing failed"
            })

    successful = [
        result for result in results
        if "error" not in result
    ]

    flooded = [
        result for result in successful
        if result["flooded"]
    ]

    return {
        "success": True,
        "total_locations": len(pairs),
        "processed_locations": len(successful),
        "failed_locations": len(pairs) - len(successful),
        "flooded_locations": len(flooded),
        "results": results
    }