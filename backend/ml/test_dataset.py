
import os
from dataset import AquaWatchDataset
from torch.utils.data import DataLoader

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

CSV_PATH = os.path.join(
    BASE_DIR, "datasets", "ml", "pairs.csv"
)

dataset = AquaWatchDataset(CSV_PATH)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

print("AQUAWATCH DATASET TEST")
print("----------------------")

print("Total image pairs:", len(dataset))

for batch in loader:

    before = batch["before_image"]
    after = batch["after_image"]
    labels = batch["label"]

    print("Before shape:", before.shape)
    print("After shape:", after.shape)
    print("Labels:", labels)
    print("Locations:", batch["location"])

print("Dataset loaded successfully!")
