
import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AquaWatchDataset(Dataset):

    def __init__(self, csv_file, root_dir=None, transform=None):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir or os.path.dirname(csv_file)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Check required CSV columns
        required_columns = [
            "location",
            "before_image",
            "after_image",
            "flooded"
        ]

        for column in required_columns:
            if column not in self.data.columns:
                raise ValueError(
                    f"Missing required column: {column}"
                )

        # Check labels
        if self.data["flooded"].isna().any():
            raise ValueError("Some flood labels are missing.")

        if not self.data["flooded"].isin([0, 1]).all():
            raise ValueError("Flood labels must be 0 or 1.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        before_path = os.path.join(
            self.root_dir,
            row["before_image"]
        )

        after_path = os.path.join(
            self.root_dir,
            row["after_image"]
        )

        # Check image files
        if not os.path.exists(before_path):
            raise FileNotFoundError(
                f"Before image not found: {before_path}"
            )

        if not os.path.exists(after_path):
            raise FileNotFoundError(
                f"After image not found: {after_path}"
            )

        # Load images
        before_image = Image.open(
            before_path
        ).convert("RGB")

        after_image = Image.open(
            after_path
        ).convert("RGB")

        # Apply transformations
        before_image = self.transform(before_image)
        after_image = self.transform(after_image)

        # Flood label
        label = torch.tensor(
            int(row["flooded"]),
            dtype=torch.float32
        )

        return {
            "before_image": before_image,
            "after_image": after_image,
            "label": label,
            "location": row["location"]
        }
