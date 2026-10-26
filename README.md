# Chest X-Ray Disease Detection

**Author**: Dhanush Nagineni ([dhanushnaginane@gmail.com](mailto:dhanushnaginane@gmail.com))

React + FastAPI thesis project for chest X-ray screening support.

The app analyzes a chest X-ray image and shows:

- Final Result
- Model Results
- Image Evidence / Heatmaps
- Medical Information Assistant

This project is for academic screening support. It does not provide a medical diagnosis.

## Dataset Source

**Dataset taken from Kaggle**: [Chest X-Ray (COVID-19 & Pneumonia & Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)  
**URL**: `https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis`  
Contains chest radiograph images labeled across the four target categories:
- COVID-19
- Normal
- Pneumonia
- Tuberculosis

## Project Overview

Frontend:

- React + Vite
- Runs on port `5173` (or `3000` via Docker)
- Direct S3 image archive viewer & real-time prediction display

Backend:

- FastAPI + PyTorch + TorchXRayVision
- Runs on port `8000`
- Integrated with **AWS S3** and **[Floci](https://github.com/floci-io/floci)** (open-source local AWS emulator)
- Automatically archives incoming chest X-rays to an S3 bucket before triggering multi-model inference and Grad-CAM generation

Main model:

- `Chest X-Ray Multi-Disease Classifier`
- TorchXRayVision DenseNet-121 backbone
- Four classes:
  - `COVID19`
  - `NORMAL`
  - `PNEUMONIA`
  - `TUBERCULOSIS`

Training result:

- Validation accuracy: `94.74%`
- Test accuracy: `92.22%`

## Model Files

Place the main model files in:

```text
backend/models/
```

Required main model files:

```text
backend/models/torchxrayvision_densenet121_4class.pth
backend/models/torchxrayvision_densenet121_4class.metadata.json
backend/models/torchxrayvision_densenet121_4class.labels.json
```

Specialist model files are also used by the backend and can be placed in the project root or `backend/models/`:

```text
resnet18_covid.pth
resnet18_pneumonia_v2.pth
resnet18_tb.pth
```

Do not rename the model files unless you also update the backend code.

## Installation

Install backend dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

If `npm` is not installed, install Node.js from the official Node.js website or use a local Node runtime if one is already included in `frontend/.nodeenv-full`.

## Run Everything

### On Windows:
Double-click `run_app.bat` or run in terminal:
```cmd
run_app.bat
```

### On Linux / macOS:
```bash
./run_app.sh
```

The script:

- stops old processes on ports `8000` and `5173`
- starts the FastAPI backend
- starts the Vite frontend
- prints the browser URL

Open:

```text
http://127.0.0.1:5173
```

The script also prints a network URL for a phone on the same Wi-Fi, for example:

```text
http://192.168.1.3:5173
```

## Run Backend Manually

From the project root:

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Check backend health:

```text
http://127.0.0.1:8000/api/health
```

Useful backend endpoints:

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/api/models
http://127.0.0.1:8000/docs
```

## Run Frontend Manually

From the frontend folder:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

If using the local Node runtime:

```bash
cd frontend
env PATH="$PWD/.nodeenv-full/bin:$PATH" .nodeenv-full/bin/npm run dev
```

## How Prediction Works

1. The user uploads a chest X-ray image.
2. The backend checks whether the image looks like a chest radiograph.
3. The image is preprocessed for the model.
4. The Chest X-Ray Multi-Disease Classifier predicts probabilities for the four classes.
5. The app creates a Final Result:
   - `Normal`
   - `Possible COVID-19`
   - `Possible Pneumonia`
   - `Possible Tuberculosis`
   - `Uncertain Result`
6. The app shows heatmaps to indicate image regions that influenced the model output.

Screening results should be reviewed by a qualified professional.

## Medical Information Assistant

The chatbot is local and rule-based.

It can explain:

- current screening result
- COVID-19
- Pneumonia
- Tuberculosis
- Normal result
- chest X-ray terms
- symptoms in general
- general treatment categories
- confidence score
- heatmaps / Grad-CAM
- manual review

It refuses personal diagnosis and treatment-plan questions.

## Testing Images

Place test images into:

```text
test_images/
  covid/
  normal/
  pneumonia/
  tuberculosis/
```

Run:

```bash
python3 testing/test_images.py --test-dir test_images
```

The script prints the predicted label, confidence, correct label, correctness, and a confusion matrix for the current multi-disease classifier.

## Troubleshooting

### Port 8000 already in use

Use the one-command startup script:

```bash
./run_app.sh
```

It stops old processes on port `8000` before starting the backend.

Manual option:

```bash
lsof -ti tcp:8000
kill <PID>
```

### Frontend server not running

Start the frontend:

```bash
cd frontend
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

### Model not found

Check that these files exist:

```text
backend/models/torchxrayvision_densenet121_4class.pth
backend/models/torchxrayvision_densenet121_4class.metadata.json
backend/models/torchxrayvision_densenet121_4class.labels.json
```

Then restart the backend.

### npm not found

Install Node.js/npm, then run:

```bash
cd frontend
npm install
npm run dev
```

If using the local runtime:

```bash
cd frontend
env PATH="$PWD/.nodeenv-full/bin:$PATH" .nodeenv-full/bin/npm install
env PATH="$PWD/.nodeenv-full/bin:$PATH" .nodeenv-full/bin/npm run dev
```

---

## Author

- **Author**: Dhanush Nagineni
- **Email**: [dhanushnaginane@gmail.com](mailto:dhanushnaginane@gmail.com)

