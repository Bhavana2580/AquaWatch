
import numpy as np
from skimage.filters import threshold_otsu


def detect_water(image, method="fixed", threshold=-15):

    """
    Detect water pixels from Sentinel-1 SAR imagery.

    Water pixels = 1
    Non-water pixels = 0
    """

    # Identify valid pixels
    valid = np.isfinite(image)

    if not np.any(valid):
        raise ValueError("No valid pixels found in image")

    # Select thresholding method
    if method == "fixed":

        actual_threshold = threshold

    elif method == "otsu":

        actual_threshold = threshold_otsu(
            image[valid]
        )

    else:

        raise ValueError(
            "Method must be 'fixed' or 'otsu'"
        )

    # Create water mask
    water_mask = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    water_mask[valid] = (
        image[valid] < actual_threshold
    ).astype(np.uint8)

    return water_mask, float(actual_threshold)
