
import numpy as np
import rasterio


def remove_permanent_water(flood_mask, reference_path):

    with rasterio.open(reference_path) as src:
        permanent_water = src.read(1)

        if permanent_water.shape != flood_mask.shape:
            raise ValueError(
                "Reference water raster must be aligned "
                "with the flood raster."
            )

        valid = src.read_masks(1) > 0

    # Assuming 1 represents permanent water
    permanent_mask = (permanent_water == 1) & valid

    # Remove permanent water from flood detection
    actual_flood = flood_mask & (~permanent_mask)

    print("Permanent water pixels removed:",
          int((flood_mask & permanent_mask).sum()))

    return actual_flood
