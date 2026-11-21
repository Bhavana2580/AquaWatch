
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import AquaWatchDataset
from .model import SiameseResNet18


# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_DIR / "datasets" / "ml" / "pairs.csv"

MODEL_PATH = PROJECT_DIR / "models" / "aquawatch_model.pth"


def main():

    # Device selection
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("AQUAWATCH MODEL TRAINING")
    print("------------------------")
    print("Device:", device)

    # Load dataset
    dataset = AquaWatchDataset(
        csv_file=str(CSV_PATH)
    )

    print("Total image pairs:", len(dataset))

    # Check labels
    labels = dataset.data["flooded"].astype(int).tolist()

    print("Flood labels:", labels)

    if len(set(labels)) < 2:
        print(
            "WARNING: Only one class is available. "
            "This is a pipeline demonstration, "
            "not a reliable binary classifier."
        )

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    # Initialize model
    model = SiameseResNet18(
        num_classes=2,
        pretrained=False
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    EPOCHS = 15

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for batch in loader:

            before = batch["before_image"].to(device)
            after = batch["after_image"].to(device)

            labels = batch["label"].long().to(device)

            optimizer.zero_grad()

            outputs = model(before, after)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f}"
        )

    # Save model
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "SiameseResNet18",
            "num_classes": 2,
            "image_size": 224
        },
        MODEL_PATH
    )

    print("\nTraining completed!")
    print("Model saved successfully!")
    print("Location:", MODEL_PATH)


if __name__ == "__main__":
    main()
