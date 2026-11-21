
import json
from datetime import datetime
from pathlib import Path


# Project directories
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

REPORTS_DIR = PROJECT_DIR / "reports"


def generate_report(area_data, threshold, village_data=None):

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "project": "AquaWatch",
        "generated_at": datetime.now().isoformat(),
        "threshold_db": round(float(threshold), 3),
        "flood_statistics": area_data,
        "affected_villages": village_data or []
    }

    output_path = REPORTS_DIR / "flood_report.json"

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str
        )

    print("Report generated:", output_path)

    return report


if __name__ == "__main__":

    sample_area = {
        "flood_area_sqkm": 12.5,
        "flooded_pixels": 125000
    }

    sample_villages = []

    generate_report(
        sample_area,
        -15.0,
        sample_villages
    )
