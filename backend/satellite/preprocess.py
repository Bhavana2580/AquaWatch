
import rasterio
import os


def validate_satellite_images(before_path, after_path):

    if not os.path.exists(before_path):
        raise FileNotFoundError(before_path)

    if not os.path.exists(after_path):
        raise FileNotFoundError(after_path)

    with rasterio.open(before_path) as before:
        before_info = {
            "shape": before.shape,
            "crs": before.crs,
            "transform": before.transform
        }

    with rasterio.open(after_path) as after:
        after_info = {
            "shape": after.shape,
            "crs": after.crs,
            "transform": after.transform
        }

    if before_info["shape"] != after_info["shape"]:
        raise ValueError("Image dimensions do not match.")

    if before_info["crs"] != after_info["crs"]:
        raise ValueError("Coordinate systems do not match.")

    if before_info["transform"] != after_info["transform"]:
        raise ValueError("Images are not spatially aligned.")

    print("Both satellite images are ready!")

    return True
