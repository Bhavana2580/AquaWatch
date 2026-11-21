

import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .model import SiameseResNet18
from .preprocess import validate_satellite_images


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_DIR / "models" / "aquawatch_model.pth"

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# IMAGE TRANSFORMATION
# Keep this consistent with training preprocessing.
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --------------------------------------------------
# FLOOD PREDICTOR
# --------------------------------------------------

class FloodPredictor:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. "
                "Run the training script first."
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )

        self.model = SiameseResNet18(
            num_classes=2,
            pretrained=False,
        ).to(device)

        # Support the checkpoint format used during training.
        if "state_dict" not in checkpoint:
            raise ValueError(
                "Invalid model checkpoint: state_dict missing."
            )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model.eval()

    # --------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------

    def load_image(self, image_path):

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = transform(image)

            return image_tensor.unsqueeze(0).to(device)

        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(
                f"Unable to read satellite image: {error}"
            ) from error

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    @torch.no_grad()
    def predict(self, before_path, after_path):

        # Validate the image pair.
        validate_satellite_images(
            before_path,
            after_path,
        )

        before = self.load_image(before_path)
        after = self.load_image(after_path)

        outputs = self.model(before, after)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        flood_probability = probabilities[0, 1].item()

        return {
            "location_status": (
                "Flood Detected"
                if predicted_class == 1
                else "No Flood Detected"
            ),
            "flooded": predicted_class == 1,
            "flood_probability": round(
                flood_probability * 100,
                2,
            ),
            "device": str(device),
            "prediction_note": (
                "Experimental prediction only. "
                "Not validated for real-world use."
            ),
        }


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

_predictor = None


def predict_flood(before_path, after_path):

    global _predictor

    if _predictor is None:
        _predictor = FloodPredictor()

    return _predictor.predict(
        before_path,
        after_path,
    )


# --------------------------------------------------
# COMMAND LINE TEST
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python -m backend.ml.predict "
            "<before_image> <after_image>"
        )
        sys.exit(1)

    try:
        result = predict_flood(
            sys.argv[1],
            sys.argv[2],
        )

        print("\nAQUAWATCH FLOOD PREDICTION")
        print("--------------------------")

        for key, value in result.items():
            print(f"{key}: {value}")

    except Exception as error:
        print(f"Prediction failed: {error}")
        sys.exit(1)
