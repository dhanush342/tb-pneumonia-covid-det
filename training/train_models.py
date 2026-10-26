import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CANONICAL_CLASSES = ("COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FOLDER_ALIASES = {
    "covid": "COVID19",
    "covid19": "COVID19",
    "covid-19": "COVID19",
    "normal": "NORMAL",
    "pneumonia": "PNEUMONIA",
    "tb": "TUBERCULOSIS",
    "tuberculosis": "TUBERCULOSIS",
}

MODEL_CONFIGS = {
    "multidisease": {
        "id": "resnet50-multidisease",
        "display_name": "Chest X-Ray Multi-Disease Classifier",
        "architecture": "resnet50",
        "filename": "resnet50_multidisease.pth",
        "class_labels": ("COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"),
        "positive_label": None,
        "task": "multi_class",
    },
    "covid": {
        "id": "covid-resnet18",
        "display_name": "COVID-19 Specialist",
        "architecture": "resnet18",
        "filename": "resnet18_covid.pth",
        "class_labels": ("NOT_COVID19", "COVID19"),
        "positive_label": "COVID19",
        "task": "one_vs_rest",
    },
    "pneumonia": {
        "id": "pneumonia-resnet18",
        "display_name": "Pneumonia Specialist",
        "architecture": "resnet18",
        "filename": "resnet18_pneumonia_v2.pth",
        "class_labels": ("NOT_PNEUMONIA", "PNEUMONIA"),
        "positive_label": "PNEUMONIA",
        "task": "one_vs_rest",
    },
    "tuberculosis": {
        "id": "tb-resnet18",
        "display_name": "Tuberculosis Specialist",
        "architecture": "resnet18",
        "filename": "resnet18_tb.pth",
        "class_labels": ("NOT_TUBERCULOSIS", "TUBERCULOSIS"),
        "positive_label": "TUBERCULOSIS",
        "task": "one_vs_rest",
    },
}

PREPROCESSING = {
    "input_size": [224, 224],
    "color": "grayscale converted to 3 RGB channels",
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "train_augmentation": [
        "RandomResizedCrop(224, scale=(0.90, 1.00), ratio=(0.95, 1.05))",
        "RandomRotation(7)",
        "ColorJitter(brightness=0.08, contrast=0.08)",
    ],
}


def canonical_label(folder_name):
    key = folder_name.lower().replace("_", "-").strip()
    if key not in FOLDER_ALIASES:
        raise ValueError(f"Unknown class folder '{folder_name}'. Expected covid, normal, pneumonia, tuberculosis.")
    return FOLDER_ALIASES[key]


def image_paths(root):
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


class XrayFolderDataset(Dataset):
    def __init__(self, root, config, transform):
        self.root = Path(root)
        self.config = config
        self.transform = transform
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        for class_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            source_label = canonical_label(class_dir.name)
            for image_path in image_paths(class_dir):
                target = self.target_index(source_label)
                self.samples.append((image_path, target, source_label))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def target_index(self, source_label):
        labels = self.config["class_labels"]
        positive_label = self.config["positive_label"]
        if positive_label:
            return labels.index(positive_label) if source_label == positive_label else 0
        return labels.index(source_label)

    def class_counts(self):
        return Counter(target for _path, target, _source_label in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target, _source_label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), target


def train_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(224, scale=(0.90, 1.00), ratio=(0.95, 1.05)),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(PREPROCESSING["normalization_mean"], PREPROCESSING["normalization_std"]),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(PREPROCESSING["normalization_mean"], PREPROCESSING["normalization_std"]),
        ]
    )


def build_model(config, pretrained):
    num_classes = len(config["class_labels"])
    if config["architecture"] == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def class_weights(dataset, num_classes, device):
    counts = dataset.class_counts()
    total = sum(counts.values())
    weights = []
    for index in range(num_classes):
        weights.append(total / max(counts.get(index, 1), 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_matrix(labels, predictions, num_classes):
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for actual, predicted in zip(labels, predictions):
        matrix[actual][predicted] += 1
    return matrix


def evaluate(model, loader, device, num_classes):
    model.eval()
    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            predicted_labels.extend(predictions)
            actual_labels.extend(labels.tolist())

    correct = sum(1 for actual, predicted in zip(actual_labels, predicted_labels) if actual == predicted)
    accuracy = correct / max(len(actual_labels), 1)
    return accuracy, confusion_matrix(actual_labels, predicted_labels, num_classes)


def train_one_model(name, config, args, device):
    train_root = Path(args.data_dir) / "train"
    val_root = Path(args.data_dir) / "val"

    train_dataset = XrayFolderDataset(train_root, config, train_transform())
    val_dataset = XrayFolderDataset(val_root, config, eval_transform())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(config, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset, len(config["class_labels"]), device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        val_accuracy, val_matrix = evaluate(model, val_loader, device, len(config["class_labels"]))
        epoch_loss = running_loss / len(train_dataset)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "validation_accuracy": val_accuracy})
        print(f"{name} epoch {epoch}/{args.epochs} - loss {epoch_loss:.4f} - val_acc {val_accuracy:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / config["filename"]
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_id": config["id"],
        "display_name": config["display_name"],
        "architecture": config["architecture"],
        "weight_file": config["filename"],
        "class_labels": list(config["class_labels"]),
        "positive_label": config["positive_label"],
        "task": config["task"],
        "preprocessing": PREPROCESSING,
        "training_date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "validation_accuracy": history[-1]["validation_accuracy"],
        "confusion_matrix": {
            "labels": list(config["class_labels"]),
            "matrix": val_matrix,
        },
        "history": history,
    }
    metadata_path = output_dir / f"{Path(config['filename']).stem}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Saved {metadata_path}")


def selected_models(selection):
    if selection == "all":
        return ("multidisease", "covid", "pneumonia", "tuberculosis")
    if selection == "specialists":
        return ("covid", "pneumonia", "tuberculosis")
    return (selection,)


def parse_args():
    parser = argparse.ArgumentParser(description="Train chest X-ray multi-disease and one-vs-rest specialist models.")
    parser.add_argument("--data-dir", default="dataset", help="Folder with train/ and val/ subfolders.")
    parser.add_argument("--output-dir", default=".", help="Where to save .pth and .metadata.json files.")
    parser.add_argument("--models", default="all", choices=["all", "multidisease", "specialists", "covid", "pneumonia", "tuberculosis"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained initialization.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    for model_name in selected_models(args.models):
        train_one_model(model_name, MODEL_CONFIGS[model_name], args, device)


if __name__ == "__main__":
    main()
