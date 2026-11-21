
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


def analyze_villages(flood_mask, raster_path, village_path):

    villages = gpd.read_file(village_path)

    with rasterio.open(raster_path) as src:

        villages = villages.to_crs(src.crs)

        transform = src.transform
        shape = flood_mask.shape

        results = []

        for _, village in villages.iterrows():

            geometry = village.geometry

            inside = geometry_mask(
                [geometry],
                out_shape=shape,
                transform=transform,
                invert=True
            )

            village_pixels = np.count_nonzero(inside)

            flooded_pixels = np.count_nonzero(
                inside & flood_mask
            )

            if village_pixels == 0:
                continue

            percentage = (
                flooded_pixels / village_pixels
            ) * 100

            results.append({
                "village": str(
                    village.get("name", "Unknown")
                ),
                "flooded_pixels": int(flooded_pixels),
                "total_pixels": int(village_pixels),
                "flood_percentage": round(percentage, 2)
            })

    results.sort(
        key=lambda x: x["flood_percentage"],
        reverse=True
    )

    return results
