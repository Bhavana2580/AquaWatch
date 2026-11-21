
import torch
import torch.nn as nn
from torchvision import models


class SiameseResNet18(nn.Module):
    """
    Siamese ResNet18 for before-and-after
    satellite image comparison.

    Both images share the same feature extractor.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        # Load ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet18(weights=weights)

        # Shared feature extractor
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # ResNet18 produces 512 features per image.
        # Absolute difference captures changes.
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, before, after):

        # Extract features using shared encoder
        before_features = self.encoder(before)
        after_features = self.encoder(after)

        # Flatten feature vectors
        before_features = torch.flatten(
            before_features, 1
        )

        after_features = torch.flatten(
            after_features, 1
        )

        # Calculate feature difference
        difference = torch.abs(
            before_features - after_features
        )

        # Classification
        output = self.classifier(difference)

        return output


# --------------------------------------------------
# MODEL TEST
# --------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cpu")

    model = SiameseResNet18(
        num_classes=2,
        pretrained=False
    ).to(device)

    model.eval()

    # Test using dummy satellite images
    before = torch.randn(2, 3, 224, 224)
    after = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        predictions = model(before, after)

    print("\nAQUAWATCH MODEL TEST")
    print("--------------------")
    print("Device:", device)
    print("Before image shape:", before.shape)
    print("After image shape:", after.shape)
    print("Output shape:", predictions.shape)
    print("Model initialized successfully!")
