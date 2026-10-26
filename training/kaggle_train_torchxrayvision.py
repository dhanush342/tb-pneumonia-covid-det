import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchxrayvision as xrv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_LABELS = ("COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_BASENAME = "torchxrayvision_densenet121_4class"

FOLDER_ALIASES = {
    "covid": "COVID19",
    "covid19": "COVID19",
    "covid-19": "COVID19",
    "normal": "NORMAL",
    "pneumonia": "PNEUMONIA",
    "tb": "TUBERCULOSIS",
    "tuberculosis": "TUBERCULOSIS",
}

PREPROCESSING = {
    "library": "torchxrayvision",
    "input_size": [224, 224],
    "color": "grayscale single-channel chest X-ray",
    "pixel_scaling": "xrv.datasets.normalize(image, 255), approximately [-1024, 1024]",
    "resize": "xrv.datasets.XRayCenterCrop() then xrv.datasets.XRayResizer(224)",
    "train_augmentation": [
        "RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.94, 1.04))",
        "ColorJitter(brightness=0.08, contrast=0.08)",
    ],
}


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    return str(kaggle_working if kaggle_working.exists() else Path("."))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonical_label(folder_name):
    key = folder_name.lower().replace("_", "-").strip()
    if key not in FOLDER_ALIASES:
        raise ValueError(f"Unknown class folder '{folder_name}'. Use covid, normal, pneumonia, or tuberculosis.")
    return FOLDER_ALIASES[key]


def image_paths(root):
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


class XRayVisionFolderDataset(Dataset):
    def __init__(self, root, training=False):
        self.root = Path(root)
        self.training = training
        self.samples = []
        self.label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
        self.xrv_transform = transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224),
            ]
        )
        self.augmentation = transforms.Compose(
            [
                transforms.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.94, 1.04), fill=0),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        )

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        for class_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            label = canonical_label(class_dir.name)
            for image_path in image_paths(class_dir):
                self.samples.append((image_path, self.label_to_index[label], label))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def class_counts(self):
        return Counter(target for _path, target, _label in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target, _label = self.samples[index]
        image = Image.open(image_path).convert("L")
        if self.training:
            image = self.augmentation(image)

        image_array = np.array(image).astype(np.float32)
        image_array = xrv.datasets.normalize(image_array, 255)
        image_array = image_array[None, :, :]
        image_array = self.xrv_transform(image_array)
        image_tensor = torch.from_numpy(image_array).float()
        return image_tensor, target


def build_model(weights_name, num_classes):
    model = xrv.models.DenseNet(weights=weights_name)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.targets = list(CLASS_LABELS)
    model.pathologies = model.targets
    model.apply_sigmoid = False
    model.op_threshs = None
    return model


def class_weights(dataset, device):
    counts = dataset.class_counts()
    total = sum(counts.values())
    weights = [total / max(counts.get(index, 1), 1) for index in range(len(CLASS_LABELS))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_matrix(actual_labels, predicted_labels):
    matrix = [[0 for _ in CLASS_LABELS] for _ in CLASS_LABELS]
    for actual, predicted in zip(actual_labels, predicted_labels):
        matrix[actual][predicted] += 1
    return matrix


def evaluate(model, loader, device):
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
    return accuracy, confusion_matrix(actual_labels, predicted_labels)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    train_dataset = XRayVisionFolderDataset(data_dir / "train", training=True)
    val_dataset = XRayVisionFolderDataset(data_dir / "val", training=False)
    test_dataset = XRayVisionFolderDataset(data_dir / "test", training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.weights, len(CLASS_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_dataset, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_validation_accuracy = -1.0
    best_state = None
    best_val_matrix = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        val_accuracy, val_matrix = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_accuracy": val_accuracy,
            }
        )
        print(f"Epoch {epoch}/{args.epochs} - loss {train_loss:.4f} - val_acc {val_accuracy:.4f}")

        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_val_matrix = val_matrix
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")

    model.load_state_dict(best_state)
    model.to(device)
    test_accuracy, test_matrix = evaluate(model, test_loader, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / f"{OUTPUT_BASENAME}.pth"
    metadata_path = output_dir / f"{OUTPUT_BASENAME}.metadata.json"
    labels_path = output_dir / f"{OUTPUT_BASENAME}.labels.json"

    torch.save(best_state, weights_path)

    class_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    label_metadata = {
        "class_labels": list(CLASS_LABELS),
        "class_to_index": class_to_index,
        "index_to_class": {str(index): label for label, index in class_to_index.items()},
    }
    metadata = {
        "model_id": "torchxrayvision-densenet121-4class",
        "display_name": "Chest X-Ray Multi-Disease Classifier",
        "architecture": "TorchXRayVision DenseNet-121",
        "base_weights": args.weights,
        "weight_file": weights_path.name,
        "class_labels": list(CLASS_LABELS),
        "preprocessing": PREPROCESSING,
        "training_date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "validation_accuracy": best_validation_accuracy,
        "test_accuracy": test_accuracy,
        "validation_confusion_matrix": {
            "labels": list(CLASS_LABELS),
            "matrix": best_val_matrix,
        },
        "test_confusion_matrix": {
            "labels": list(CLASS_LABELS),
            "matrix": test_matrix,
        },
        "history": history,
        "important_note": (
            "TorchXRayVision pretrained outputs are not the same as this thesis class list. "
            "This file is valid for the thesis classes only after fine-tuning."
        ),
    }

    labels_path.write_text(json.dumps(label_metadata, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved weights: {weights_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved label mapping: {labels_path}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a TorchXRayVision DenseNet-121 model on four thesis classes.")
    parser.add_argument("--data-dir", default="dataset", help="Folder containing dataset/train, dataset/val, and dataset/test.")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Where Kaggle should save model artifacts.")
    parser.add_argument("--weights", default="densenet121-res224-all", help="TorchXRayVision pretrained weights name.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
