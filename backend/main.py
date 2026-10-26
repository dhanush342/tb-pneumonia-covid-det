from io import BytesIO
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .inference import get_model_catalog, get_runtime_status, predict_image, validate_chest_xray_image
from .s3_service import get_s3_status, upload_xray_to_s3


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chest X-Ray Disease Detection API",
    description="FastAPI backend for chest X-ray classifier with AWS S3 / Floci storage.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Chest X-Ray Disease Detection API",
        "health": "/api/health",
        "models": "/api/models",
        "prediction": "/api/predict",
        "s3_status": "/api/s3/status",
    }


@app.get("/api/health")
def health():
    return get_runtime_status()


@app.get("/api/models")
def models():
    return {"models": get_model_catalog()}


@app.get("/api/s3/status")
def s3_status():
    return get_s3_status()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file could not be read as an image.") from exc

    # 1. Upload to AWS S3 bucket (or Floci emulator when running locally)
    s3_meta = upload_xray_to_s3(
        file_bytes=contents,
        filename=file.filename or "uploaded-xray.png",
        content_type=file.content_type or "image/png",
    )

    try:
        # 2. Validate chest X-ray
        validation = validate_chest_xray_image(image)
        if not validation.accepted:
            payload = validation.to_payload()
            payload["s3"] = s3_meta
            raise HTTPException(status_code=422, detail=payload)

        # 3. Perform AI multi-model disease detection & Grad-CAM heatmap generation
        result = predict_image(image, filename=file.filename or "uploaded-image", validation=validation)
        result["s3"] = s3_meta
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for uploaded file: %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail={
                "title": "Analysis failed",
                "message": f"Backend prediction error: {exc}",
                "stage": "prediction",
                "s3": s3_meta,
            },
        ) from exc

