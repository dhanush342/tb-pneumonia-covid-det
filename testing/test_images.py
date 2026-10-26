import argparse
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import predict_model_label


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


def canonical_label(folder_name):
    key = folder_name.lower().replace("_", "-").strip()
    if key not in FOLDER_ALIASES:
        raise ValueError(f"Unknown test folder '{folder_name}'. Expected covid, normal, pneumonia, tuberculosis.")
    return FOLDER_ALIASES[key]


def image_paths(root):
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def empty_matrix():
    return [[0 for _ in CANONICAL_CLASSES] for _ in CANONICAL_CLASSES]


def print_matrix(matrix):
    print("\nChest X-Ray Multi-Disease Classifier confusion matrix")
    print("actual \\ predicted".ljust(20) + " ".join(label[:11].rjust(12) for label in CANONICAL_CLASSES))
    for actual_index, actual_label in enumerate(CANONICAL_CLASSES):
        values = " ".join(str(value).rjust(12) for value in matrix[actual_index])
        print(actual_label.ljust(20) + values)


def main():
    parser = argparse.ArgumentParser(description="Test the chest X-ray multi-disease classifier on test_images folders.")
    parser.add_argument("--test-dir", default="test_images")
    args = parser.parse_args()

    test_root = Path(args.test_dir)
    if not test_root.exists():
        raise FileNotFoundError(f"Test folder not found: {test_root}")

    matrix = empty_matrix()
    total = 0
    correct = 0

    for class_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        expected_label = canonical_label(class_dir.name)
        for image_path in image_paths(class_dir):
            image = Image.open(image_path).convert("RGB")
            prediction = predict_model_label(image)
            predicted_label = prediction["label"]
            is_correct = predicted_label == expected_label
            total += 1
            correct += int(is_correct)

            actual_index = CANONICAL_CLASSES.index(expected_label)
            predicted_index = CANONICAL_CLASSES.index(predicted_label)
            matrix[actual_index][predicted_index] += 1

            print(
                f"{image_path} | predicted={prediction['labelText']} "
                f"confidence={prediction['confidence']}% | correct_label={expected_label} | correct={is_correct}"
            )

    accuracy = correct / max(total, 1)
    print(f"\nTotal: {total}  Correct: {correct}  Accuracy: {accuracy:.4f}")
    print_matrix(matrix)


if __name__ == "__main__":
    main()
