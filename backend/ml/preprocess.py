
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import json


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "ml"
CSV_PATH = DATASET_DIR / "pairs.csv"

OUTPUT_DIR = DATASET_DIR / "processed"

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# VALIDATE SATELLITE IMAGE PAIR
# --------------------------------------------------

def validate_satellite_images(before_path, after_path):

    before_path = Path(before_path)
    after_path = Path(after_path)

    if not before_path.exists():
        raise FileNotFoundError(
            f"Before image not found: {before_path}"
        )

    if not after_path.exists():
        raise FileNotFoundError(
            f"After image not found: {after_path}"
        )

    for image_path in [before_path, after_path]:

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                if img.width < 32 or img.height < 32:
                    raise ValueError(
                        f"Image resolution too small: {image_path}"
                    )

        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(
                f"Invalid image file: {image_path}"
            ) from error

    return True


# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as img:

        img = ImageOps.exif_transpose(img)

        original_size = img.size

        img = img.convert("RGB")

        img = ImageOps.fit(
            img,
            IMAGE_SIZE,
            method=Image.Resampling.BILINEAR
        )

        image_array = np.asarray(
            img,
            dtype=np.float32
        ) / 255.0

    return image_array, original_size


# --------------------------------------------------
# PROCESS DATASET
# --------------------------------------------------

def preprocess_dataset():

    print("\n========== AQUAWATCH PREPROCESSING ==========\n")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"pairs.csv not found: {CSV_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(CSV_PATH)

    required_columns = [
        "location",
        "after_image",
        "before_image"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing CSV column: {column}"
            )

    processed_count = 0
    errors = []

    for index, row in df.iterrows():

        location = str(row["location"]).strip()

        print(f"\nProcessing {location}...")

        try:

            if not location or location.lower() == "nan":
                raise ValueError("Location name is empty")

            after_path = DATASET_DIR / str(
                row["after_image"]
            ).strip()

            before_path = DATASET_DIR / str(
                row["before_image"]
            ).strip()

            validate_satellite_images(
                before_path,
                after_path
            )

            after_array, after_size = load_image(
                after_path
            )

            before_array, before_size = load_image(
                before_path
            )

            print("Before image size:", before_size)
            print("After image size:", after_size)

            if before_size != after_size:
                print(
                    "WARNING: Original image dimensions differ."
                )

            # Avoid unsafe location strings in filenames.
            safe_location = "".join(
                char if char.isalnum() or char in "-_"
                else "_"
                for char in location
            )

            output_file = OUTPUT_DIR / f"{safe_location}.npz"

            np.savez_compressed(
                output_file,
                before=before_array,
                after=after_array,
                location=location
            )

            processed_count += 1

            print(
                f"Successfully processed: {location}"
            )

        except Exception as error:

            errors.append({
                "location": location,
                "error": str(error)
            })

            print(
                f"ERROR processing {location}: {error}"
            )

    # Save processing report.
    report = {
        "total_pairs": len(df),
        "processed_pairs": processed_count,
        "failed_pairs": len(errors),
        "image_size": IMAGE_SIZE,
        "errors": errors
    }

    report_path = OUTPUT_DIR / "report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== PROCESSING COMPLETE ==========")
    print("Total pairs:", len(df))
    print("Successfully processed:", processed_count)
    print("Failed pairs:", len(errors))
    print("Output folder:", OUTPUT_DIR)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    preprocess_dataset()
