
import numpy as np

from flood_detection.water_detection import detect_water


# Sample SAR backscatter values in dB

image = np.array([
    [-22, -19, -10],
    [-18, -12, -5],
    [-25, -8, -3]
], dtype=np.float32)


# Detect water

water_mask, threshold = detect_water(
    image,
    method="fixed",
    threshold=-15
)


print("Original Image:")
print(image)

print("\nWater Mask:")
print(water_mask)

print("\nThreshold:", threshold, "dB")
