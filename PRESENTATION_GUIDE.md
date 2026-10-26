# Presentation Guide

## Project Aim

This project is a chest X-ray screening support application. It uses a trained deep learning model to classify an uploaded chest X-ray into one of four classes:

- COVID-19
- Normal
- Pneumonia
- Tuberculosis

The goal is to help demonstrate how medical image classification can be combined with a clean web interface, confidence scores, heatmaps, and safe educational explanations.

## App Workflow

1. The user opens the React frontend.
2. The user uploads a chest X-ray image.
3. The FastAPI backend checks whether the image looks like a chest radiograph.
4. The backend preprocesses the image.
5. The model predicts class scores for the supported disease classes.
6. The app shows the Final Result, Model Results, and Image Evidence.
7. The Medical Information Assistant can explain the result and basic X-ray terms.

## Model Used

The main model is called:

```text
Chest X-Ray Multi-Disease Classifier
```

It is based on:

```text
TorchXRayVision DenseNet-121
```

The model was fine-tuned for this thesis project to classify four classes:

```text
COVID-19, Normal, Pneumonia, Tuberculosis
```

## Training Result

Validation accuracy:

```text
94.74%
```

Test accuracy:

```text
92.22%
```

During the presentation, say that these results show the model performed well on the prepared dataset, but the app is still a screening support tool and not a medical diagnosis system.

## How Prediction Works

Simple explanation:

1. The image is converted into the format expected by the model.
2. The model compares the image against four learned classes.
3. The model outputs one probability score for each class.
4. The class with the strongest score becomes the main prediction.
5. If the disease confidence is high enough, the app shows `Possible COVID-19`, `Possible Pneumonia`, or `Possible Tuberculosis`.
6. If Normal is strongest, the app shows `Normal`.
7. If confidence is weak, the app shows `Uncertain Result`.

## What Confidence Means

Confidence is the model's score for the selected class.

Example:

```text
Pneumonia: 92%
```

This means the model strongly selected Pneumonia compared with the other classes. It does not mean the patient definitely has pneumonia.

Good presentation sentence:

```text
Confidence shows how strongly the model selected one class, but it does not guarantee medical correctness.
```

## What Heatmaps Mean

The heatmap shows image regions that influenced the model output.

It helps answer:

```text
Which parts of the X-ray affected the model prediction?
```

Important limitation:

```text
A heatmap is supporting visual evidence, not proof of disease.
```

## Chatbot Purpose

The Medical Information Assistant is local and rule-based.

It helps users understand:

- the current screening result
- confidence score
- heatmaps
- COVID-19
- Pneumonia
- Tuberculosis
- Normal result
- common chest X-ray terms
- why manual review is needed

It does not use OpenAI, Hugging Face, or any external API.

It refuses personal diagnosis or treatment-plan questions with this answer:

```text
I cannot provide a personal diagnosis or treatment plan. This assistant gives general educational information only. Please consult a qualified medical professional for medical decisions.
```

## Limitations

Mention these clearly:

- The app is not a medical diagnosis system.
- The model was trained on a limited dataset.
- Results can be affected by image quality.
- Cropped, rotated, side-view, low-contrast, or internet-sourced images can reduce reliability.
- The model only detects four classes.
- A doctor must review medical decisions.

## How To Run The Demo

From the project root:

```bash
./run_app.sh
```

Open:

```text
http://127.0.0.1:5173
```

Check backend health:

```text
http://127.0.0.1:8000/api/health
```

The health endpoint should show:

```text
torchxrayvision_model_loaded: true
```

## What To Say During Presentation

Suggested short script:

```text
This application is a chest X-ray screening support tool built with React and FastAPI.
The backend uses a TorchXRayVision DenseNet-121 model fine-tuned for four classes: COVID-19, Normal, Pneumonia, and Tuberculosis.
After uploading an X-ray, the system validates the image, runs the model, and returns a final screening result with confidence information.
The heatmap shows which image regions influenced the model output.
The chatbot is local and rule-based. It explains results and medical imaging terms, but it refuses diagnosis or treatment advice.
The validation accuracy was 94.74%, and the test accuracy was 92.22%.
This is not a replacement for a doctor. It is an academic system for demonstrating AI-assisted chest X-ray screening.
```

## Demo Checklist

Before presenting:

- Start the app with `./run_app.sh`.
- Open `http://127.0.0.1:5173`.
- Check `http://127.0.0.1:8000/api/health`.
- Upload one good chest X-ray.
- Confirm the Final Result appears.
- Confirm Heatmaps appear.
- Click at least two chatbot quick buttons.
- Try one refusal question, such as `Do I have pneumonia?`
