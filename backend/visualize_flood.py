
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import folium

from pathlib import Path
from rasterio.features import shapes
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import shape, mapping


# ---------------------------------------
# PROJECT PATHS
# ---------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

PROCESSED_DIR = PROJECT_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Expected processed raster filenames
BEFORE_IMAGE = PROCESSED_DIR / "before_flood.tif"
AFTER_IMAGE = PROCESSED_DIR / "after_flood.tif"
FLOOD_MASK = PROCESSED_DIR / "flood_mask.tif"


# ---------------------------------------
# READ RASTER DATA
# ---------------------------------------

def read_raster(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Raster file not found: {path}"
        )

    with rasterio.open(path) as src:

        image = src.read(1).astype(np.float32)

        valid = (
            (src.read_masks(1) > 0)
            & np.isfinite(image)
        )

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    return image, valid, profile, transform, crs, bounds


# ---------------------------------------
# CREATE BEFORE-AFTER COMPARISON
# ---------------------------------------

def create_comparison_map():

    print("Generating flood comparison image...")

    before, before_valid, _, _, _, _ = read_raster(
        BEFORE_IMAGE
    )

    after, after_valid, _, _, _, _ = read_raster(
        AFTER_IMAGE
    )

    flood_mask, mask_valid, _, _, _, _ = read_raster(
        FLOOD_MASK
    )

    # Check image dimensions
    if not (
        before.shape == after.shape == flood_mask.shape
    ):
        raise ValueError(
            "Before image, after image and flood mask "
            "must have identical dimensions."
        )

    # Create valid flood mask
    valid = before_valid & after_valid & mask_valid

    flooded = (flood_mask == 1) & valid

    # Mask invalid pixels
    before_display = np.ma.masked_where(
        ~valid, before
    )

    after_display = np.ma.masked_where(
        ~valid, after
    )

    flood_display = np.ma.masked_where(
        ~valid, flooded.astype(float)
    )

    # Create visualization
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7)
    )

    axes[0].imshow(
        before_display,
        cmap="gray",
        vmin=-30,
        vmax=0
    )

    axes[0].set_title("Before Flood")

    axes[1].imshow(
        after_display,
        cmap="gray",
        vmin=-30,
        vmax=0
    )

    axes[1].set_title("After Flood")

    axes[2].imshow(
        before_display,
        cmap="gray",
        vmin=-30,
        vmax=0
    )

    axes[2].imshow(
        np.ma.masked_where(
            ~flooded, flooded.astype(float)
        ),
        cmap="Reds",
        alpha=0.7,
        vmin=0,
        vmax=1
    )

    axes[2].set_title("Detected Flood Areas")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "AquaWatch - Satellite Flood Analysis",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = REPORTS_DIR / "flood_comparison.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Comparison saved: {output_path}")

    return str(output_path)


# ---------------------------------------
# CREATE INTERACTIVE FLOOD MAP
# ---------------------------------------

def create_interactive_map():

    print("Generating interactive flood map...")

    flood_mask, mask_valid, _, transform, crs, bounds = (
        read_raster(FLOOD_MASK)
    )

    if crs is None:
        raise ValueError(
            "Flood mask has no CRS information."
        )

    # Identify flooded pixels
    flooded = (
        (flood_mask == 1)
        & mask_valid
    )

    if not np.any(flooded):
        raise ValueError(
            "No flooded pixels found in the flood mask."
        )

    # Convert raster bounds to latitude and longitude
    west, south, east, north = transform_bounds(
        crs,
        "EPSG:4326",
        *bounds
    )

    center_lat = (south + north) / 2
    center_lon = (west + east) / 2

    # Create interactive map
    flood_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    # Convert flooded pixels into geographic polygons
    # Each polygon represents a connected flooded region.
    polygon_count = 0

    for geom, value in shapes(
        flooded.astype(np.uint8),
        mask=flooded,
        transform=transform
    ):

        if value != 1:
            continue

        # Transform polygon into WGS84 coordinates
        geographic_geom = transform_geom(
            crs,
            "EPSG:4326",
            geom
        )

        # Simplify polygon to reduce map size
        simplified = shape(
            geographic_geom
        ).simplify(
            0.00005,
            preserve_topology=True
        )

        folium.GeoJson(
            mapping(simplified),
            style_function=lambda feature: {
                "fillColor": "#ff3333",
                "color": "#b30000",
                "weight": 1,
                "fillOpacity": 0.55
            },
            tooltip="Detected Flood Area"
        ).add_to(flood_map)

        polygon_count += 1

    # Add legend
    legend = """
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 1px solid grey;
        border-radius: 8px;
        font-size: 13px;
    ">
        <b>AquaWatch Flood Map</b><br>
        <span style="color:#ff3333;">■</span>
        Detected Flood Area
    </div>
    """

    flood_map.get_root().html.add_child(
        folium.Element(legend)
    )

    # Save interactive map
    output_path = REPORTS_DIR / "flood_map.html"

    flood_map.save(str(output_path))

    print(f"Interactive map saved: {output_path}")
    print(f"Flood regions displayed: {polygon_count}")

    return str(output_path)


# ---------------------------------------
# RUN BOTH VISUALIZATIONS
# ---------------------------------------

if __name__ == "__main__":

    create_comparison_map()

    create_interactive_map()

    print("\nAquaWatch visualizations completed!")
