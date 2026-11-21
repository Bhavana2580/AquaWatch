
import os
import numpy as np
import rasterio
from skimage.filters import threshold_otsu


def compare_flood_images(before_path, after_path, output_path):
    """
    Detect newly flooded pixels by comparing
    pre-flood and post-flood water masks.
    """

    with rasterio.open(before_path) as before_src:
        before = before_src.read(1).astype(np.float32)
        before_valid = before_src.read_masks(1) > 0
        profile = before_src.profile.copy()
        before_transform = before_src.transform
        before_crs = before_src.crs

    with rasterio.open(after_path) as after_src:
        after = after_src.read(1).astype(np.float32)
        after_valid = after_src.read_masks(1) > 0

        if (after.shape != before.shape or
                after_src.transform != before_transform or
                after_src.crs != before_crs):
            raise ValueError(
                "Before and after images must have identical "
                "dimensions, CRS and transform. Align them in SNAP."
            )

    valid = (
        before_valid & after_valid &
        np.isfinite(before) & np.isfinite(after)
    )

    if not np.any(valid):
        raise ValueError("No valid overlapping pixels found.")

    # Calculate one common threshold using both images
    combined = np.concatenate((before[valid], after[valid]))

    if np.ptp(combined) == 0:
        raise ValueError("Satellite image has no intensity variation.")

    threshold = float(threshold_otsu(combined))

    # Water detection
    water_before = (before < threshold) & valid
    water_after = (after < threshold) & valid

    # Newly flooded areas
    flood_mask = water_after & (~water_before)

    # Save flood mask
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(flood_mask.astype(np.uint8), 1)

    print("Flood comparison completed!")
    print("Threshold:", round(threshold, 3), "dB")
    print("Newly flooded pixels:", int(flood_mask.sum()))

    return flood_mask, threshold
