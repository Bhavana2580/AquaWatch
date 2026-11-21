
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Read Sentinel-1 processed image
with rasterio.open("datasets/after_flood/after_flood.tif") as src:
    image = src.read(1).astype(np.float32)
    valid = (src.read_masks(1) > 0) & np.isfinite(image)

# Extract valid pixel values
pixels = image[valid]

# Calculate automatic threshold
threshold = threshold_otsu(pixels)

print("Calculated Threshold:", threshold, "dB")

# Classify water pixels
water_mask = np.zeros(image.shape, dtype=np.uint8)
water_mask[valid] = (image[valid] < threshold).astype(np.uint8)

# Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(pixels, bins=100)
plt.axvline(threshold, color="red", linestyle="--",
            label=f"Threshold = {threshold:.2f} dB")
plt.xlabel("Backscatter (dB)")
plt.ylabel("Pixel count")
plt.title("SAR Backscatter Histogram")
plt.legend()

plt.subplot(1, 2, 2)
plt.imshow(water_mask, cmap="Blues")
plt.title("Detected Water Regions")
plt.axis("off")

plt.tight_layout()
plt.show()
