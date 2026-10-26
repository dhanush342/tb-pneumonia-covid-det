import base64
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "backend" / "models"
DEVICE = torch.device("cpu")
TORCHXRAYVISION_MODEL_ID = "torchxrayvision-densenet121-4class"
EXPECTED_FOUR_CLASS_LABELS = ("COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    filename: str
    architecture: str
    labels: Tuple[str, ...]
    role: str
    focus: str
    description: str
    positive_label: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None
    preprocessing: str = "imagenet_rgb"
    required: bool = True
    decision_note: Optional[str] = None


@dataclass
class RuntimeModel:
    spec: ModelSpec
    model: nn.Module
    target_layer: nn.Module
    lock: Lock
    model_path: Path
    labels_path: Optional[Path]
    metadata_path: Optional[Path]


@dataclass(frozen=True)
class ImageValidationResult:
    accepted: bool
    score: int
    required_score: int
    title: str
    message: str
    checks: Tuple[Dict[str, object], ...]

    def to_payload(self) -> Dict[str, object]:
        return {
            "accepted": self.accepted,
            "score": self.score,
            "requiredScore": self.required_score,
            "title": self.title,
            "message": self.message,
            "checks": list(self.checks),
        }


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        id=TORCHXRAYVISION_MODEL_ID,
        display_name="Chest X-Ray Multi-Disease Classifier",
        filename="torchxrayvision_densenet121_4class.pth",
        architecture="TorchXRayVision DenseNet-121",
        labels=EXPECTED_FOUR_CLASS_LABELS,
        role="Main 4-class chest X-ray classifier",
        focus="COVID-19, normal, pneumonia, tuberculosis",
        description=(
            "Fine-tuned TorchXRayVision DenseNet-121 model trained on the thesis classes. "
            "This is the main 4-class classifier when the Kaggle-trained weights are present."
        ),
        preprocessing="torchxrayvision",
        required=False,
        decision_note="Main 4-class classifier",
    ),
    ModelSpec(
        id="covid-resnet18",
        display_name="COVID-19 Specialist",
        filename="resnet18_covid.pth",
        architecture="ResNet-18",
        labels=("COVID19", "NORMAL"),
        role="Binary disease check",
        focus="COVID-19",
        description="Specialist model that compares COVID-19 against normal X-rays.",
        positive_label="COVID19",
    ),
    ModelSpec(
        id="pneumonia-resnet18",
        display_name="Pneumonia Specialist",
        filename="resnet18_pneumonia_v2.pth",
        architecture="ResNet-18",
        labels=("NORMAL", "PNEUMONIA"),
        role="Binary disease check",
        focus="Pneumonia",
        description="Specialist model that compares pneumonia against normal X-rays.",
        positive_label="PNEUMONIA",
    ),
    ModelSpec(
        id="tb-resnet18",
        display_name="Tuberculosis Specialist",
        filename="resnet18_tb.pth",
        architecture="ResNet-18",
        labels=("NORMAL", "TUBERCULOSIS"),
        role="Binary disease check",
        focus="Tuberculosis",
        description="Specialist model that compares tuberculosis against normal X-rays.",
        positive_label="TUBERCULOSIS",
    ),
)


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _state_dict_from_file(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
    return checkpoint


def _remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


MODEL_LOAD_STATUS: Dict[str, Dict[str, object]] = {}


def _log_model(message: str) -> None:
    print(f"[model-loader] {message}", flush=True)


def _candidate_dirs_for(spec: ModelSpec) -> Tuple[Path, ...]:
    if spec.id == TORCHXRAYVISION_MODEL_ID:
        return (MODEL_DIR, BASE_DIR)
    return (BASE_DIR, MODEL_DIR)


def _resolve_model_path(spec: ModelSpec) -> Optional[Path]:
    for directory in _candidate_dirs_for(spec):
        path = directory / spec.filename
        if path.exists():
            return path
    return None


def _sidecar_path(model_path: Path, suffix: str) -> Optional[Path]:
    path = model_path.with_name(f"{model_path.stem}.{suffix}.json")
    return path if path.exists() else None


def _load_json_file(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _labels_from_labels_file(labels_data: Optional[Dict[str, object]]) -> Optional[Tuple[str, ...]]:
    if not labels_data:
        return None
    if "class_labels" in labels_data:
        return tuple(str(label) for label in labels_data["class_labels"])
    if "index_to_class" in labels_data:
        index_to_class = labels_data["index_to_class"]
        return tuple(index_to_class[str(index)] for index in range(len(index_to_class)))
    return None


def _spec_with_sidecars(
    spec: ModelSpec,
    labels_path: Optional[Path],
    metadata_path: Optional[Path],
) -> ModelSpec:
    labels_data = _load_json_file(labels_path)
    metadata = _load_json_file(metadata_path)

    labels = _labels_from_labels_file(labels_data) or tuple((metadata or {}).get("class_labels") or spec.labels)
    positive_label = (metadata or {}).get("positive_label", spec.positive_label)

    if spec.id == TORCHXRAYVISION_MODEL_ID and labels != EXPECTED_FOUR_CLASS_LABELS:
        raise ValueError(
            "TorchXRayVision label order mismatch. "
            f"Expected {EXPECTED_FOUR_CLASS_LABELS}, found {labels}."
        )

    if metadata and spec.id == TORCHXRAYVISION_MODEL_ID:
        metadata = {**metadata, "display_name": spec.display_name}

    return replace(spec, labels=labels, positive_label=positive_label, metadata=metadata)


def _build_model(spec: ModelSpec, model_path: Path) -> Tuple[nn.Module, nn.Module]:
    if spec.architecture == "ResNet-18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(spec.labels))
        target_layer = model.layer4[-1]
    elif spec.architecture == "TorchXRayVision DenseNet-121":
        if xrv is None:
            raise RuntimeError("TorchXRayVision is not installed.")
        model = xrv.models.DenseNet(weights=None, num_classes=len(spec.labels))
        model.targets = list(spec.labels)
        model.pathologies = list(spec.labels)
        model.apply_sigmoid = False
        model.op_threshs = None
        target_layer = model.features.denseblock4
    else:
        raise ValueError(f"Unsupported architecture: {spec.architecture}")

    state_dict = _remove_module_prefix(_state_dict_from_file(model_path))
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, target_layer


def _load_models() -> List[RuntimeModel]:
    loaded = []
    for base_spec in MODEL_SPECS:
        model_path = _resolve_model_path(base_spec)
        status = {
            "loaded": False,
            "model_path": str(model_path) if model_path else None,
            "labels_path": None,
            "metadata_path": None,
            "labels": list(base_spec.labels),
            "error": None,
        }
        MODEL_LOAD_STATUS[base_spec.id] = status

        if model_path is None:
            if base_spec.required:
                raise FileNotFoundError(f"Required model file not found: {base_spec.filename}")
            _log_model(f"SKIPPED optional model: {base_spec.display_name}")
            _log_model(f"Missing model file: {base_spec.filename}")
            continue

        labels_path = _sidecar_path(model_path, "labels")
        metadata_path = _sidecar_path(model_path, "metadata")
        status["labels_path"] = str(labels_path) if labels_path else None
        status["metadata_path"] = str(metadata_path) if metadata_path else None

        try:
            spec = _spec_with_sidecars(base_spec, labels_path, metadata_path)
            status["labels"] = list(spec.labels)
            _log_model(f"Loading: {spec.display_name}")
            _log_model(f"Model file: {model_path}")
            _log_model(f"Labels file: {labels_path if labels_path else 'not found'}")
            _log_model(f"Metadata file: {metadata_path if metadata_path else 'not found'}")
            _log_model(f"Labels: {', '.join(spec.labels)}")

            model, target_layer = _build_model(spec, model_path)
        except Exception as exc:
            status["error"] = str(exc)
            if base_spec.required:
                raise
            _log_model(f"FAILED optional model: {base_spec.display_name}")
            _log_model(f"Error: {exc}")
            continue

        status["loaded"] = True
        if spec.id == TORCHXRAYVISION_MODEL_ID:
            _log_model("TorchXRayVision model loading succeeded: true")
        loaded.append(
            RuntimeModel(
                spec=spec,
                model=model,
                target_layer=target_layer,
                lock=Lock(),
                model_path=model_path,
                labels_path=labels_path,
                metadata_path=metadata_path,
            )
        )

    if not MODEL_LOAD_STATUS.get(TORCHXRAYVISION_MODEL_ID, {}).get("loaded"):
        _log_model("TorchXRayVision model loading succeeded: false")
    return loaded


RUNTIME_MODELS = _load_models()

SPECIALIST_MODEL_IDS = ("covid-resnet18", "pneumonia-resnet18", "tb-resnet18")
SPECIALIST_DISEASE_THRESHOLD = 0.80
BORDERLINE_THRESHOLD = 0.60


def _check(name: str, passed: bool, value: object, expected: str) -> Dict[str, object]:
    return {
        "name": name,
        "passed": passed,
        "value": value,
        "expected": expected,
    }


def _confidence_ratio(model_result: Dict[str, object]) -> float:
    return float(model_result["confidence"]) / 100


def _colorfulness(rgb: np.ndarray) -> float:
    rgb_float = rgb.astype(np.float32)
    red, green, blue = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    return float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values = left.astype(np.float32).flatten()
    right_values = right.astype(np.float32).flatten()
    if float(np.std(left_values)) < 1e-4 or float(np.std(right_values)) < 1e-4:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def validate_chest_xray_image(image: Image.Image) -> ImageValidationResult:
    """Reject obvious non-chest-X-ray uploads before running disease inference.

    This is a conservative heuristic gate. A dedicated chest-radiograph detector
    would be stronger, but these checks stop common random photos from reaching
    the disease classifiers.
    """

    width, height = image.size
    rgb = np.array(image.convert("RGB").resize((256, 256)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    min_side = min(width, height)
    aspect_ratio = width / max(height, 1)
    color_score = _colorfulness(rgb)
    mean_saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
    colorful_pixels = float(np.mean(hsv[:, :, 1] > 55))
    gray_std = float(np.std(gray))

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edge_density = float(np.mean(edges > 0))

    thorax = gray[34:204, 28:228]
    left_lung_area = thorax[:, 10:88]
    right_lung_area = np.fliplr(thorax[:, 112:190])
    center_spine_area = thorax[:, 82:118]
    lung_area = np.concatenate([left_lung_area, right_lung_area], axis=1)

    symmetry = _safe_correlation(left_lung_area, right_lung_area)
    left_mean = float(np.mean(left_lung_area))
    right_mean = float(np.mean(right_lung_area))
    center_mean = float(np.mean(center_spine_area))
    lung_mean = float(np.mean(lung_area))
    lung_balance = abs(left_mean - right_mean)
    center_to_lung_delta = center_mean - lung_mean

    checks = (
        _check("Image size", min_side >= 160, f"{width} x {height}px", "at least 160px on the shortest side"),
        _check("Image shape", 0.55 <= aspect_ratio <= 1.65, round(aspect_ratio, 2), "portrait or square-like radiograph"),
        _check(
            "Mostly grayscale",
            color_score <= 28 and mean_saturation <= 0.18 and colorful_pixels <= 0.12,
            {
                "colorfulness": round(color_score, 2),
                "meanSaturation": round(mean_saturation, 3),
                "colorfulPixels": round(colorful_pixels, 3),
            },
            "low color saturation like a radiograph",
        ),
        _check("Radiograph contrast", 28 <= gray_std <= 92, round(gray_std, 2), "visible X-ray contrast, not flat or photographic"),
        _check("Anatomy edge pattern", 0.025 <= edge_density <= 0.24, round(edge_density, 3), "moderate chest/rib edge density"),
        _check(
            "Left/right lung symmetry",
            symmetry >= 0.2 and lung_balance <= 38,
            {"symmetry": round(symmetry, 3), "leftRightMeanDifference": round(lung_balance, 2)},
            "two similar lung-field regions",
        ),
        _check(
            "Central thorax pattern",
            center_to_lung_delta >= 2,
            round(center_to_lung_delta, 2),
            "brighter central thorax than surrounding lung fields",
        ),
    )

    score = sum(1 for item in checks if item["passed"])
    required_score = 6
    accepted = bool(checks[0]["passed"] and checks[2]["passed"] and score >= required_score)

    if accepted:
        title = "Chest X-ray validation passed"
        message = "The image looks like a chest radiograph, so the analysis can continue."
    else:
        title = "Upload a chest X-ray"
        message = (
            "This image does not look enough like a frontal chest/lung X-ray. "
            "Please upload a grayscale chest radiograph before running the disease models."
        )

    return ImageValidationResult(
        accepted=accepted,
        score=score,
        required_score=required_score,
        title=title,
        message=message,
        checks=checks,
    )


def get_model_catalog() -> List[Dict[str, object]]:
    return [
        {
            "id": runtime.spec.id,
            "name": runtime.spec.display_name,
            "architecture": runtime.spec.architecture,
            "role": runtime.spec.role,
            "focus": runtime.spec.focus,
            "labels": list(runtime.spec.labels),
            "positiveLabel": runtime.spec.positive_label,
            "metadata": runtime.spec.metadata,
            "description": runtime.spec.description,
            "modelPath": str(runtime.model_path),
            "labelsPath": str(runtime.labels_path) if runtime.labels_path else None,
            "metadataPath": str(runtime.metadata_path) if runtime.metadata_path else None,
        }
        for runtime in RUNTIME_MODELS
    ]


def get_runtime_status() -> Dict[str, object]:
    xrv_status = MODEL_LOAD_STATUS.get(TORCHXRAYVISION_MODEL_ID, {})
    return {
        "status": "ok",
        "modelsLoaded": len(RUNTIME_MODELS),
        "torchxrayvision_model_loaded": bool(xrv_status.get("loaded")),
        "model_path": xrv_status.get("model_path"),
        "labels_path": xrv_status.get("labels_path"),
        "metadata_path": xrv_status.get("metadata_path"),
        "labels": xrv_status.get("labels") or [],
        "model_load_status": MODEL_LOAD_STATUS,
    }


def _generate_gradcam(runtime: RuntimeModel, image_tensor: torch.Tensor) -> Tuple[np.ndarray, int, List[float]]:
    gradients = None
    activations = None

    def save_activation(_module, _inputs, output):
        nonlocal activations
        activations = output.detach()

    def save_gradient(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    forward_handle = runtime.target_layer.register_forward_hook(save_activation)
    backward_handle = runtime.target_layer.register_full_backward_hook(save_gradient)

    try:
        runtime.model.zero_grad(set_to_none=True)
        output = runtime.model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        class_idx = int(torch.argmax(probabilities).item())
        output[0, class_idx].backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model activations.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probabilities.detach().cpu().tolist()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _overlay_heatmap(image: Image.Image, cam: np.ndarray) -> str:
    base = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.62 * base + 0.38 * heatmap, 0, 1)
    overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))

    buffer = io.BytesIO()
    overlay_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _torchxrayvision_tensor(image: Image.Image) -> torch.Tensor:
    if xrv is None:
        raise RuntimeError("TorchXRayVision is required for this model but is not installed.")

    gray = np.array(image.convert("L")).astype(np.float32)
    normalized = xrv.datasets.normalize(gray, 255)
    normalized = normalized[None, :, :]
    transform = transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )
    tensor = torch.from_numpy(transform(normalized)).float()
    return tensor.unsqueeze(0).to(DEVICE)


def _image_tensor_for_spec(spec: ModelSpec, image: Image.Image) -> torch.Tensor:
    if spec.preprocessing == "torchxrayvision":
        return _torchxrayvision_tensor(image)
    return IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def _probabilities(spec: ModelSpec, raw_probabilities: List[float]) -> List[Dict[str, object]]:
    return [
        {
            "label": label,
            "value": round(float(value) * 100, 2),
            "kind": "finding" if _is_positive_label(spec, label) else "normal",
        }
        for label, value in zip(spec.labels, raw_probabilities)
    ]


def _is_positive_label(spec: ModelSpec, label: str) -> bool:
    if spec.positive_label:
        return label == spec.positive_label
    return label != "NORMAL"


def _signal_status(positive: bool, confidence_ratio: float) -> Tuple[str, Optional[str]]:
    if not positive:
        return "Normal leaning", None
    if confidence_ratio >= SPECIALIST_DISEASE_THRESHOLD:
        return "Confirmed disease signal", None
    if confidence_ratio >= BORDERLINE_THRESHOLD:
        return "Borderline signal", "Borderline signal - not confirmed"
    return "Weak signal", None


def _model_result(runtime: RuntimeModel, image: Image.Image, image_tensor: Optional[torch.Tensor]) -> Dict[str, object]:
    image_tensor = _image_tensor_for_spec(runtime.spec, image) if image_tensor is None else image_tensor
    with runtime.lock:
        cam, class_idx, raw_probabilities = _generate_gradcam(runtime, image_tensor.clone())

    spec = runtime.spec
    probabilities = _probabilities(spec, raw_probabilities)
    prediction = spec.labels[class_idx]
    confidence = probabilities[class_idx]["value"]
    positive = _is_positive_label(spec, prediction)
    confidence_ratio = float(confidence) / 100
    signal_status, signal_note = _signal_status(positive, confidence_ratio)
    disease_probability = max((item["value"] for item in probabilities if item["kind"] == "finding"), default=0)

    return {
        "id": spec.id,
        "name": spec.display_name,
        "architecture": spec.architecture,
        "role": spec.role,
        "focus": spec.focus,
        "description": spec.description,
        "labels": list(spec.labels),
        "positiveLabel": spec.positive_label,
        "metadata": spec.metadata,
        "decisionNote": spec.decision_note,
        "prediction": prediction,
        "confidence": confidence,
        "confidenceRatio": round(confidence_ratio, 4),
        "status": "finding" if positive else "normal",
        "statusLabel": signal_status,
        "signalNote": signal_note,
        "confirmedDiseaseSignal": positive and confidence_ratio >= SPECIALIST_DISEASE_THRESHOLD,
        "borderlineSignal": positive and BORDERLINE_THRESHOLD <= confidence_ratio < SPECIALIST_DISEASE_THRESHOLD,
        "weakSignal": positive and confidence_ratio < BORDERLINE_THRESHOLD,
        "diseaseProbability": round(float(disease_probability), 2),
        "probabilities": probabilities,
        "heatmap": _overlay_heatmap(image, cam),
    }


def _format_label(label: str) -> str:
    if label == "COVID19":
        return "COVID-19"
    if label.startswith("NOT_"):
        return "Not " + _format_label(label[4:])
    if label == "UNCERTAIN_REVIEW_NEEDED":
        return "Uncertain Result"
    return label.title()


def _model_by_id(model_results: List[Dict[str, object]], model_id: str) -> Dict[str, object]:
    return next(item for item in model_results if item["id"] == model_id)


def _strong_disease_signals(specialists: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        item
        for item in specialists
        if item["status"] == "finding" and _confidence_ratio(item) >= SPECIALIST_DISEASE_THRESHOLD
    ]


def _borderline_disease_signals(specialists: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        item
        for item in specialists
        if item["status"] == "finding" and BORDERLINE_THRESHOLD <= _confidence_ratio(item) < SPECIALIST_DISEASE_THRESHOLD
    ]


def _torchxrayvision_decision(main_model: Dict[str, object]) -> Dict[str, object]:
    """Create the final screening result from the primary 4-class model."""

    prediction = str(main_model["prediction"])
    confidence = float(main_model["confidence"])
    confidence_ratio = confidence / 100
    model_name = str(main_model["name"])

    if prediction == "NORMAL":
        return {
            "result": "NORMAL",
            "resultLabel": "Normal",
            "confidence": confidence,
            "confidenceStatus": "Normal was the strongest class",
            "reason": f"{model_name} predicted Normal with {confidence:.2f}% confidence.",
            "recommendation": "Screening results should be reviewed by a qualified professional.",
            "warning": None,
            "severity": "low",
            "sourceModelId": TORCHXRAYVISION_MODEL_ID,
            "sourceModelName": model_name,
        }

    disease = _format_label(prediction)
    if confidence_ratio >= SPECIALIST_DISEASE_THRESHOLD:
        return {
            "result": prediction,
            "resultLabel": f"Possible {disease}",
            "confidence": confidence,
            "confidenceStatus": "One disease class produced strong confidence",
            "reason": (
                f"{model_name} predicted {disease} with {confidence:.2f}% confidence, "
                "which is above the 80% threshold."
            ),
            "recommendation": "Screening results should be reviewed by a qualified professional.",
            "warning": None,
            "severity": "elevated",
            "sourceModelId": TORCHXRAYVISION_MODEL_ID,
            "sourceModelName": model_name,
        }

    return {
        "result": "UNCERTAIN_REVIEW_NEEDED",
        "resultLabel": "Uncertain Result",
        "confidence": confidence,
        "confidenceStatus": "Weak model confidence",
        "reason": (
            f"{model_name} predicted {disease}, but confidence was only {confidence:.2f}%, "
            "which is below the 80% disease threshold."
        ),
        "recommendation": "Screening results should be reviewed by a qualified professional.",
        "warning": None,
        "severity": "review",
        "sourceModelId": TORCHXRAYVISION_MODEL_ID,
        "sourceModelName": model_name,
    }


def make_final_ai_decision(model_results: List[Dict[str, object]]) -> Dict[str, object]:
    main_models = [item for item in model_results if item["id"] == TORCHXRAYVISION_MODEL_ID]
    if main_models:
        return _torchxrayvision_decision(main_models[0])

    specialists = [item for item in model_results if item["id"] in SPECIALIST_MODEL_IDS]
    borderline_signals = _borderline_disease_signals(specialists)
    strong_signals = _strong_disease_signals(specialists)

    if not strong_signals:
        all_specialists_normal = all(item["status"] == "normal" for item in specialists)
        if all_specialists_normal:
            return {
                "result": "NORMAL",
                "resultLabel": "Normal",
                "confidence": None,
                "confidenceStatus": "No specialist disease signal detected",
                "reason": "COVID-19, Pneumonia, and Tuberculosis specialists all predicted Normal.",
                "recommendation": "Screening results should be reviewed by a qualified professional.",
                "warning": None,
                "severity": "low",
                "sourceModelId": None,
                "sourceModelName": "Specialist fallback logic",
            }

        return {
            "result": "UNCERTAIN_REVIEW_NEEDED",
            "resultLabel": "Uncertain Result",
            "confidence": None,
            "confidenceStatus": "Weak or borderline specialist confidence",
            "reason": "A specialist predicted disease, but no specialist reached the 80% disease threshold.",
            "recommendation": "Screening results should be reviewed by a qualified professional.",
            "warning": None,
            "severity": "review",
            "sourceModelId": None,
            "sourceModelName": "Specialist fallback logic",
        }

    if len(strong_signals) == 1:
        specialist = strong_signals[0]
        disease = _format_label(str(specialist["prediction"]))
        return {
            "result": str(specialist["prediction"]),
            "resultLabel": f"Possible {disease}",
            "confidence": float(specialist["confidence"]),
            "confidenceStatus": "One specialist produced a strong disease signal",
            "reason": (
                "Only the matching specialist detected disease above the confirmation threshold."
            ),
            "recommendation": "Screening results should be reviewed by a qualified professional.",
            "warning": None,
            "severity": "elevated",
            "sourceModelId": specialist["id"],
            "sourceModelName": specialist["name"],
        }

    return {
        "result": "UNCERTAIN_REVIEW_NEEDED",
        "resultLabel": "Uncertain Result",
        "confidence": None,
        "confidenceStatus": "Multiple disease signals detected",
        "reason": "More than one specialist predicted disease, so the system cannot choose one reliable final diagnosis.",
        "recommendation": "Screening results should be reviewed by a qualified professional.",
        "warning": None,
        "severity": "review",
        "sourceModelId": None,
        "sourceModelName": "Specialist fallback logic",
    }


def predict_model_label(image: Image.Image, model_id: str = TORCHXRAYVISION_MODEL_ID) -> Dict[str, object]:
    runtime = _model_by_id_runtime(model_id)
    rgb_image = image.convert("RGB")
    image_tensor = _image_tensor_for_spec(runtime.spec, rgb_image)

    with runtime.lock, torch.no_grad():
        output = runtime.model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0].detach().cpu().tolist()

    class_idx = int(np.argmax(probabilities))
    spec = runtime.spec
    label = spec.labels[class_idx]
    return {
        "modelId": spec.id,
        "label": label,
        "labelText": _format_label(label),
        "confidence": round(float(probabilities[class_idx]) * 100, 2),
        "probabilities": _probabilities(spec, probabilities),
    }


def _model_by_id_runtime(model_id: str) -> RuntimeModel:
    return next(runtime for runtime in RUNTIME_MODELS if runtime.spec.id == model_id)


def _agreement_summary(model_results: List[Dict[str, object]]) -> Dict[str, object]:
    specialists = [item for item in model_results if item["id"] in SPECIALIST_MODEL_IDS]
    positive_specialists = [item for item in specialists if item["status"] == "finding"]
    final_decision = make_final_ai_decision(model_results)

    if final_decision.get("sourceModelId") == TORCHXRAYVISION_MODEL_ID:
        agreement = "Main classifier loaded"
        severity = final_decision["severity"]
    elif final_decision["severity"] == "low":
        agreement = "No specialist disease signal detected"
        severity = "low"
    elif final_decision["severity"] == "elevated":
        agreement = final_decision["confidenceStatus"]
        severity = "elevated"
    else:
        agreement = "Multiple specialist disease signals detected"
        severity = "review"

    return {
        "finalPrediction": final_decision["result"],
        "finalResult": final_decision["resultLabel"],
        "confidence": final_decision.get("confidence"),
        "confidenceStatus": final_decision["confidenceStatus"],
        "agreement": agreement,
        "reason": final_decision["reason"],
        "recommendation": final_decision["recommendation"],
        "warning": final_decision["warning"],
        "severity": severity,
        "finalDecision": final_decision,
        "positiveSpecialists": [item["name"] for item in positive_specialists],
        "disclaimer": "Screening results should be reviewed by a qualified professional.",
    }


def predict_image(
    image: Image.Image,
    filename: str = "uploaded-image",
    validation: Optional[ImageValidationResult] = None,
) -> Dict[str, object]:
    rgb_image = image.convert("RGB")
    model_results = [_model_result(runtime, rgb_image, None) for runtime in RUNTIME_MODELS]

    return {
        "image": {
            "filename": filename,
            "width": rgb_image.width,
            "height": rgb_image.height,
        },
        "validation": validation.to_payload() if validation else None,
        "summary": _agreement_summary(model_results),
        "models": model_results,
    }
