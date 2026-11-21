
import numpy as np
import rasterio


def calculate_flood_area(flood_mask, reference_image):

    with rasterio.open(reference_image) as src:

        # Sentinel-1 images should preferably use a
        # projected CRS such as UTM.
        if src.crs is None or not src.crs.is_projected:
            raise ValueError(
                "Use a projected CRS (preferably UTM) "
                "for accurate pixel-area calculation."
            )

        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)

        pixel_area = pixel_width * pixel_height

    flooded_pixels = int(np.count_nonzero(flood_mask))

    area_square_metres = flooded_pixels * pixel_area

    area_square_km = area_square_metres / 1_000_000

    print("Flooded pixels:", flooded_pixels)
    print("Flooded area:", round(area_square_km, 3), "sq. km")

    return {
        "flooded_pixels": flooded_pixels,
        "pixel_area_sqm": pixel_area,
        "flood_area_sqm": area_square_metres,
        "flood_area_sqkm": area_square_km
    }
