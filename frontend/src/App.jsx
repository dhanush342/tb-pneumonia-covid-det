import { useEffect, useState } from "react";

const apiBaseUrl =
  import.meta.env.VITE_API_BASE_URL || `${window.location.protocol}//${window.location.hostname}:8000`;
const TORCHXRAYVISION_MODEL_ID = "torchxrayvision-densenet121-4class";
// Confidence thresholds used by the screening result and chatbot explanations.
const SPECIALIST_DISEASE_THRESHOLD = 80;
const BORDERLINE_THRESHOLD = 60;
const MEDICAL_REFUSAL =
  "I cannot provide a personal diagnosis or treatment plan. This assistant gives general educational information only. Please consult a qualified medical professional for medical decisions.";
const OFF_TOPIC_RESPONSE =
  "I can only answer questions related to chest X-rays, lung diseases detected by this system, model results, confidence scores, and X-ray terms.";
const DEFAULT_CHAT_MESSAGE =
  "Ask about the current screening result or general chest X-ray information. I can explain COVID-19, pneumonia, tuberculosis, normal results, common X-ray terms, confidence, heatmaps, and manual review.";
const QUICK_ACTIONS = [
  { id: "result", label: "Explain my result", prompt: "Explain my result" },
  { id: "pneumonia", label: "What does pneumonia mean?", prompt: "What is pneumonia?" },
  { id: "symptoms", label: "What symptoms are common?", prompt: "What symptoms are common?" },
  { id: "review", label: "Should I see a doctor?", prompt: "Why manual review?" },
  { id: "heatmap", label: "What does the heatmap show?", prompt: "What does heatmap show?" },
  { id: "covid_possible", label: "Can this be COVID-19?", prompt: "Can this be COVID-19?" }
];

const EDUCATIONAL_TOPICS = {
  covid:
    "COVID-19 is a respiratory infection caused by the SARS-CoV-2 virus. On chest X-rays, more severe cases may show patchy areas of increased density, often described as opacities, but X-rays alone cannot confirm COVID-19 without clinical history and appropriate testing.",
  covid_danger:
    "COVID-19 can range from mild to serious. In some cases, it can affect the lungs and cause breathing problems or pneumonia-like changes. Risk depends on factors such as age, health condition, and severity of infection. This is general information only.",
  pneumonia:
    "Pneumonia is an infection or inflammation of lung tissue. On a chest X-ray it may appear as an opacity, infiltrate, or consolidation. A doctor interprets these findings together with symptoms, examination, and sometimes lab tests.",
  pneumonia_serious:
    "Pneumonia can be mild or serious depending on the cause, the patient's condition, and how much of the lung is affected. It can cause cough, fever, chest discomfort, and breathing difficulty. Medical review is important.",
  tuberculosis:
    "Tuberculosis is an infection caused by Mycobacterium tuberculosis. In the lungs it can damage tissue and may produce X-ray findings such as upper-lung opacities, cavities, nodules, scarring, or areas that look abnormal. Diagnosis usually needs clinical evaluation and microbiology testing, not an X-ray alone.",
  tuberculosis_lung_damage:
    "Tuberculosis is a bacterial infection that commonly affects the lungs. It can cause inflammation, tissue damage, cavities, scarring, and long-lasting cough. Chest X-rays may show abnormal patterns, but TB must be confirmed with medical tests.",
  normal:
    "A Normal result means the model selected the normal class as the strongest output among COVID-19, Pneumonia, Tuberculosis, and Normal. It does not prove that the lungs are healthy or that every condition is absent; it only describes this model's screening output.",
  symptoms:
    "Common respiratory symptoms can include cough, fever, shortness of breath, chest discomfort, fatigue, and sputum production. Symptoms overlap across many conditions, so they cannot identify a disease by themselves.",
  covid_symptoms:
    "COVID-19 symptoms can include fever, cough, sore throat, tiredness, shortness of breath, body aches, and loss of smell or taste. Some people have mild symptoms, while others develop more serious breathing problems.",
  pneumonia_symptoms:
    "Pneumonia symptoms can include cough, fever, chills, chest pain when breathing or coughing, shortness of breath, fatigue, and sputum production. Symptoms vary by age, cause, and severity.",
  tuberculosis_symptoms:
    "Tuberculosis symptoms can include a cough lasting several weeks, fever, night sweats, weight loss, tiredness, chest pain, and sometimes coughing up blood. Symptoms can develop slowly and need medical evaluation.",
  treatment_categories:
    "Treatment categories depend on the cause and severity. In general, clinicians may consider observation, supportive care, antiviral treatment for selected viral infections, antibiotics for selected bacterial infections, tuberculosis-specific multi-drug therapy, oxygen support, or hospital care when needed. A qualified medical professional must choose any treatment plan.",
  severity:
    "Severity usually depends on symptoms, oxygen level, vital signs, medical history, physical examination, and imaging findings. A model confidence score or heatmap cannot determine clinical severity by itself.",
  long_term_lung_effects:
    "Some lung infections can cause temporary inflammation, and in more serious cases they may leave longer-lasting effects such as scarring, reduced lung function, or breathing difficulty. Not everyone has long-term damage. A medical professional would need clinical information and tests to evaluate this.",
  recovery:
    "Recovery depends on the cause, severity, age, general health, and treatment plan. Some respiratory infections improve fully, while more serious infections can take longer and may need follow-up.",
  complications:
    "Possible complications from serious lung infections can include breathing difficulty, low oxygen levels, fluid around the lungs, scarring, or reduced lung function. The risk depends on the disease and the person's overall health."
};

const XRAY_TERMS = {
  consolidation:
    "Consolidation is a chest X-ray term meaning part of the lung looks denser than normal, often because air spaces may be filled with fluid, infection, or other material. It can be seen in pneumonia, but a doctor must interpret it with symptoms and medical history.",
  opacity:
    "Opacity means an area on the X-ray looks whiter or denser than expected. It is a descriptive term, not a diagnosis. Opacities can come from infection, fluid, scarring, poor image quality, overlapping anatomy, or other causes.",
  infiltrate:
    "Infiltrate is a commonly used term for an abnormal-looking area in the lung, often an opacity. It may suggest infection or inflammation, but it is not specific enough to diagnose a disease alone.",
  lesion:
    "A lesion means an abnormal area seen in tissue or on an image. On a chest X-ray, a lesion can describe many different findings, so it needs professional interpretation and clinical context.",
  inflammation:
    "Inflammation is the body's response to irritation, infection, or injury. In the lungs it may contribute to X-ray changes such as opacities, but the exact cause needs clinical evaluation.",
  infection:
    "Infection means germs such as viruses, bacteria, or mycobacteria are affecting tissue. Chest X-rays can show patterns that may fit infection, but they do not identify the germ by themselves.",
  "abnormal shadow":
    "An abnormal shadow means an area on the X-ray looks different from expected. It is a descriptive phrase, not a diagnosis. It may be caused by infection, fluid, scarring, overlapping anatomy, image quality, or other conditions.",
  fluid:
    "Fluid on or around the lungs can make parts of a chest X-ray look denser or whiter. Fluid may be inside air spaces, around the lungs, or related to inflammation, but the exact cause needs clinical interpretation.",
  heatmap:
    "A heatmap, or Grad-CAM image, highlights regions that influenced the model output. It is supporting evidence for understanding the model's behavior, not proof that disease is present.",
  "grad-cam":
    "Grad-CAM is a heatmap method that shows image regions that influenced a model's prediction. It helps explain model behavior, but it does not prove that a disease is present.",
  "confidence score":
    "A confidence score shows how strongly the model selected one class compared with the others. High confidence does not guarantee medical correctness, and low confidence means the result should be treated cautiously.",
  confidence:
    "A confidence score shows how strongly the model selected one class compared with the others. High confidence does not guarantee medical correctness, and low confidence means the result should be treated cautiously.",
  "class probability":
    "A class probability is the model's score for one possible output class, such as COVID-19, Normal, Pneumonia, or Tuberculosis. The highest probability usually becomes the predicted class, but it is still a model estimate.",
  "false positive":
    "A false positive means the model suggests a disease class when the real case is not that disease. This can happen because of image quality, dataset bias, overlapping X-ray patterns, or model limitations.",
  "false negative":
    "A false negative means the model misses a disease that is actually present. This is one reason the app recommends manual review and should not be used as a standalone medical decision tool."
};

const CHAT_KNOWLEDGE_BASE = [
  { topic: "covid_danger", keywords: ["is covid dangerous", "covid dangerous", "is covid serious", "covid serious"] },
  { topic: "pneumonia_serious", keywords: ["is pneumonia serious", "pneumonia serious", "is pneumonia dangerous", "pneumonia dangerous"] },
  {
    topic: "tuberculosis_lung_damage",
    keywords: [
      "what tuberculosis do to your lung",
      "what tuberculosis does to the lung",
      "what tb does to the lung",
      "tb lung damage",
      "tuberculosis lung damage",
      "tuberculosis damage"
    ]
  },
  {
    topic: "long_term_lung_effects",
    keywords: [
      "dangerous for your lungs long term",
      "dangerous for lungs long term",
      "dangerous for your lungs",
      "dangerous for lungs",
      "long term lung",
      "long-term lung",
      "long term lungs",
      "long-term lungs",
      "lung damage",
      "damage lungs",
      "affect lungs",
      "harm lungs",
      "lasting lung",
      "permanent lung",
      "scarring",
      "reduced lung function"
    ]
  },
  { topic: "recovery", keywords: ["recovery", "recover", "get better", "healing"] },
  { topic: "complications", keywords: ["complication", "complications", "long lasting effects", "long-lasting effects"] },
  { topic: "covid_symptoms", keywords: ["symptoms of covid", "covid symptoms", "covid-19 symptoms", "coronavirus symptoms"] },
  { topic: "pneumonia_symptoms", keywords: ["pneumonia symptoms", "symptoms of pneumonia"] },
  { topic: "tuberculosis_symptoms", keywords: ["tuberculosis symptoms", "tb symptoms", "symptoms of tb", "symptoms of tuberculosis"] },
  { topic: "covid", keywords: ["what is covid", "what is covid-19", "covid", "covid-19", "coronavirus"] },
  { topic: "pneumonia", keywords: ["what is pneumonia", "pneumonia"] },
  {
    topic: "tuberculosis",
    keywords: ["what is tuberculosis", "what is tb", "tuberculosis", "tb", "what tuberculosis do to your lung", "what tuberculosis does to the lung"]
  },
  { topic: "normal", keywords: ["normal x-ray", "normal result", "normal mean", "what does normal mean", "what does normal"] },
  { topic: "treatment_categories", keywords: ["what treatments are usually considered", "treatment categories", "general treatment", "usual treatment"] },
  { topic: "severity", keywords: ["severity", "severe", "how serious"] }
];

function formatPrediction(label) {
  if (!label) return "Waiting for image";
  if (label === "UNCERTAIN_REVIEW_NEEDED") return "Uncertain Result";
  if (label === "NO_CONFIRMED_DISEASE_LIKELY_NORMAL") return "No confirmed disease detected - Likely Normal";
  if (label === "NORMAL") return "Normal";
  if (label === "COVID19") return "COVID-19";
  if (label.startsWith("NOT_")) return `Not ${formatPrediction(label.slice(4))}`;
  return label.charAt(0) + label.slice(1).toLowerCase();
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "not available";
  return `${number.toFixed(2).replace(".00", "")}%`;
}

function getPrimaryModel(models) {
  return models.find((model) => model.id === TORCHXRAYVISION_MODEL_ID) || models[0] || null;
}

function sortedProbabilities(model) {
  return [...(model?.probabilities || [])].sort((a, b) => Number(b.value) - Number(a.value));
}

function probabilityText(model) {
  const probabilities = model?.probabilities || [];
  if (probabilities.length === 0) return "Class probabilities are not available yet.";
  return probabilities
    .map((item) => `${formatPrediction(item.label)}: ${formatPercent(item.value)}`)
    .join(", ");
}

function isMedicalQuestion(text) {
  const normalized = text.toLowerCase();
  const blockedPhrases = [
    "do i have",
    "do they have",
    "does this mean i have",
    "what medicine",
    "which medicine",
    "medication",
    "which antibiotic",
    "what antibiotic",
    "what should i take",
    "should i take",
    "should i use",
    "how do i treat",
    "how to treat",
    "cure",
    "should i go to the hospital",
    "go to the hospital",
    "go to hospital",
    "should i go hospital",
    "do i need hospital",
    "do i need emergency",
    "should i go emergency",
    "should i go to emergency",
    "should i see a doctor",
    "definitely",
    "is this tuberculosis",
    "is this pneumonia",
    "is this covid",
    "can i ignore",
    "ignore this result",
    "diagnose",
    "diagnosis"
  ];

  return blockedPhrases.some((phrase) => normalized.includes(phrase));
}

function matchesAny(normalized, keywords) {
  return keywords.some((keyword) => normalized.includes(keyword));
}

function buildCaseContext(result, visibleModels, file) {
  const primaryModel = getPrimaryModel(visibleModels);
  const finalDecision = result?.summary?.finalDecision;
  const orderedProbabilities = sortedProbabilities(primaryModel);
  const topClass = orderedProbabilities[0];
  const secondClass = orderedProbabilities[1];

  return {
    hasResult: Boolean(result),
    filename: result?.image?.filename || file?.name || "not available",
    finalResult:
      finalDecision?.resultLabel ||
      result?.summary?.finalResult ||
      formatPrediction(result?.summary?.finalPrediction),
    confidenceStatus: finalDecision?.confidenceStatus || result?.summary?.confidenceStatus || "not available",
    reason: finalDecision?.reason || result?.summary?.reason || "No explanation is available yet.",
    recommendation:
      finalDecision?.recommendation ||
      result?.summary?.recommendation ||
      "This screening output should be reviewed by a qualified medical professional.",
    disclaimer:
      result?.summary?.disclaimer ||
      "This is a screening support result only and is not a medical diagnosis.",
    modelName: primaryModel?.name || "Primary classifier",
    predictedClass: formatPrediction(primaryModel?.prediction || finalDecision?.result),
    confidence: primaryModel?.confidence ?? finalDecision?.confidence,
    probabilities: orderedProbabilities,
    probabilitySummary: probabilityText(primaryModel),
    heatmapAvailable: Boolean(primaryModel?.heatmap),
    heatmapModels: visibleModels.filter((model) => model.heatmap).map((model) => model.name),
    topClass,
    secondClass
  };
}

function noResultResponse() {
  return "Please upload and analyze a chest X-ray first. Then I can explain the current result using the uploaded image name, predicted class, confidence score, heatmap availability, and model name.";
}

function explainResult(context) {
  if (!context.hasResult) return noResultResponse();

  const confidence = formatPercent(context.confidence);
  const thresholdText = `${SPECIALIST_DISEASE_THRESHOLD}%`;

  if (context.finalResult === "Normal") {
    return `${context.modelName} selected Normal as the strongest class for ${context.filename} with ${confidence} confidence. The final result is Normal because the Normal class was stronger than COVID-19, Pneumonia, and Tuberculosis in this screening output. ${context.disclaimer}`;
  }

  if (context.finalResult.includes("Uncertain")) {
    return `The current result for ${context.filename} is ${context.finalResult}. ${context.modelName} selected ${context.predictedClass} with ${confidence} confidence, but the app keeps the case uncertain when the output is weak, below the ${thresholdText} disease threshold, or not reliable enough for a clear screening result. ${context.disclaimer}`;
  }

  return `The current result for ${context.filename} is ${context.finalResult}. ${context.modelName} predicted ${context.predictedClass} with ${confidence} confidence. This is treated as a stronger screening signal when the selected disease class is above the ${thresholdText} decision threshold. ${context.disclaimer}`;
}

function compareProbabilities(context) {
  if (!context.hasResult) return noResultResponse();

  const top = context.topClass;
  const second = context.secondClass;
  const comparison = top
    ? `${formatPrediction(top.label)} had the strongest probability at ${formatPercent(top.value)}.`
    : "No strongest class is available yet.";
  const margin =
    top && second
      ? ` The next strongest class was ${formatPrediction(second.label)} at ${formatPercent(second.value)}.`
      : "";

  return `The model compared COVID-19, Normal, Pneumonia, and Tuberculosis. ${comparison}${margin} Current probabilities: ${context.probabilitySummary}`;
}

function explainConfidence(context) {
  const currentCase = context.hasResult
    ? ` In the current case, ${context.modelName} selected ${context.predictedClass} with ${formatPercent(context.confidence)} confidence.`
    : "";

  return `A confidence score shows how strongly the model selected one class compared with the others.${currentCase} In this app, a disease output above ${SPECIALIST_DISEASE_THRESHOLD}% is treated as a stronger screening signal, while ${BORDERLINE_THRESHOLD}% to ${SPECIALIST_DISEASE_THRESHOLD}% is treated as borderline. High confidence does not guarantee medical correctness.`;
}

function explainUncertainty(context) {
  if (!context.hasResult) {
    return "Uncertainty means the model output should be treated cautiously. This can happen when confidence is low, class probabilities are close, the image quality is limited, or the X-ray pattern is not clear.";
  }

  const top = context.topClass;
  const second = context.secondClass;
  const confidence = Number(context.confidence);
  const closeClasses = top && second && Number(top.value) - Number(second.value) <= 10;
  const reasons = [];

  if (context.finalResult.includes("Uncertain")) reasons.push("the final result is already marked for review");
  if (Number.isFinite(confidence) && confidence < SPECIALIST_DISEASE_THRESHOLD) {
    reasons.push(`the confidence is below the ${SPECIALIST_DISEASE_THRESHOLD}% threshold`);
  }
  if (closeClasses) {
    reasons.push(
      `${formatPrediction(top.label)} and ${formatPrediction(second.label)} are close in probability`
    );
  }

  if (reasons.length === 0) {
    return "This case is not currently marked as uncertain, but the output should still be reviewed as a screening result rather than treated as a diagnosis.";
  }

  return `This case should be treated as uncertain because ${reasons.join(", ")}. Manual review is recommended when confidence is low, probabilities are close, or the model output is not clear.`;
}

function explainHeatmap(context) {
  const availability = context.hasResult && context.heatmapAvailable ? "available" : "not available for the current case yet";
  const modelList =
    context.heatmapModels.length > 0 ? ` Heatmaps are shown for: ${context.heatmapModels.join(", ")}.` : "";

  return `A heatmap, or Grad-CAM image, highlights image regions that influenced the model output. It is supporting visual evidence, not proof of disease. The heatmap for ${context.modelName} is ${availability}.${modelList}`;
}

function imageQualityLimitations() {
  return (
    "Image quality can affect the screening output. Results can be less reliable with cropped X-rays, rotated images, side-view X-rays, low contrast, text labels, borders, and internet images from different sources."
  );
}

function caseSummary(context) {
  if (!context.hasResult) return noResultResponse();

  return [
    `Case summary`,
    `Image filename: ${context.filename}`,
    `Predicted class: ${context.predictedClass}`,
    `Confidence: ${formatPercent(context.confidence)}`,
    `Class probabilities: ${context.probabilitySummary}`,
    `Model used: ${context.modelName}`,
    `Heatmap: ${context.heatmapAvailable ? "available" : "not available"}`,
    `Disclaimer: ${context.disclaimer}`
  ].join("\n");
}

function explainModelScope() {
  return "The model detects four output classes: COVID-19, Normal, Pneumonia, and Tuberculosis. It does not evaluate every possible chest condition and does not replace clinical review.";
}

function explainManualReview(context) {
  if (!context.hasResult) {
    return "Manual review matters because chest X-ray patterns can overlap between conditions, image quality can affect results, and a model output is not the same as a clinical diagnosis.";
  }

  return `Manual review is recommended because this is a screening-support result, not a clinical diagnosis. Current confidence status: ${context.confidenceStatus}. Recommendation: ${context.recommendation}`;
}

function explainCovidPossibility(context) {
  const currentCase = context.hasResult
    ? ` The current screening result is ${context.finalResult}, and the model selected ${context.predictedClass} with ${formatPercent(context.confidence)} confidence.`
    : " The app can screen for COVID-19 as one of its four output classes.";

  return `COVID-19 can sometimes affect the lungs and may produce X-ray changes, but this app cannot confirm infection or replace medical testing.${currentCase} A qualified medical professional should review the result with symptoms, history, and appropriate tests.`;
}

function explainEducationalTopic(topic) {
  return EDUCATIONAL_TOPICS[topic] || explainModelScope();
}

function findKnowledgeTopic(normalized) {
  return CHAT_KNOWLEDGE_BASE.find((entry) => matchesAny(normalized, entry.keywords))?.topic || null;
}

function findXrayTerm(normalized) {
  if (normalized.includes("false positive")) return "false positive";
  if (normalized.includes("false negative")) return "false negative";
  if (normalized.includes("confidence score")) return "confidence score";
  if (normalized.includes("class probability") || normalized.includes("class probabilities")) return "class probability";
  if (normalized.includes("grad-cam") || normalized.includes("grad cam")) return "grad-cam";
  if (normalized.includes("abnormal shadow") || normalized.includes("shadow")) return "abnormal shadow";

  return Object.keys(XRAY_TERMS).find((term) => normalized.includes(term));
}

function explainTerm(term, context) {
  if (term === "heatmap") return explainHeatmap(context);
  if (term === "confidence" || term === "confidence score") return explainConfidence(context);
  return XRAY_TERMS[term] || "I can explain common chest X-ray terms such as consolidation, opacity, infiltrate, lesion, inflammation, infection, heatmap, confidence score, false positive, and false negative.";
}

function buildAssistantResponse(message, actionId, result, visibleModels, file) {
  // Local rule-based chatbot: explain project results and educational X-ray terms only.
  if (isMedicalQuestion(message)) return MEDICAL_REFUSAL;

  const context = buildCaseContext(result, visibleModels, file);
  const normalized = message.toLowerCase();

  if (actionId === "result") return explainResult(context);
  if (actionId === "pneumonia") return explainEducationalTopic("pneumonia");
  if (actionId === "symptoms") return explainEducationalTopic("symptoms");
  if (actionId === "review") return explainManualReview(context);
  if (actionId === "heatmap") return explainHeatmap(context);
  if (actionId === "covid_possible") return explainCovidPossibility(context);

  if (normalized.includes("my result") || normalized.includes("prediction") || normalized.includes("explain this result")) {
    return explainResult(context);
  }

  const knowledgeTopic = findKnowledgeTopic(normalized);
  if (knowledgeTopic) return explainEducationalTopic(knowledgeTopic);

  if (normalized.includes("symptom")) return explainEducationalTopic("symptoms");
  if (normalized.includes("compare")) return compareProbabilities(context);

  const term = findXrayTerm(normalized);
  if (term) return explainTerm(term, context);

  if (normalized.includes("probab")) {
    return compareProbabilities(context);
  }
  if (normalized.includes("threshold")) {
    return explainConfidence(context);
  }
  if (normalized.includes("uncertain") || normalized.includes("low confidence")) {
    return explainUncertainty(context);
  }
  if (normalized.includes("quality") || normalized.includes("cropped") || normalized.includes("rotated")) {
    return imageQualityLimitations();
  }
  if (normalized.includes("summary") || normalized.includes("report")) {
    return caseSummary(context);
  }

  if (normalized.includes("detect") || normalized.includes("classes") || normalized.includes("what model")) {
    return explainModelScope();
  }
  if (normalized.includes("manual review") || normalized.includes("why review")) {
    return explainManualReview(context);
  }

  return OFF_TOPIC_RESPONSE;
}

function normalizeApiError(payload) {
  if (payload?.detail && typeof payload.detail === "object") {
    return {
      title: payload.detail.title || "Analysis stopped",
      message: payload.detail.message || "The backend rejected this image.",
      stage: payload.detail.stage,
      checks: payload.detail.checks || []
    };
  }

  return {
    title: "Analysis stopped",
    message: payload?.detail || payload?.message || "The backend could not analyze this image.",
    checks: []
  };
}

async function readErrorPayload(response) {
  const rawText = await response.text().catch(() => "");
  if (!rawText) {
    return {
      detail: `Backend returned HTTP ${response.status} ${response.statusText || ""}`.trim()
    };
  }

  try {
    return JSON.parse(rawText);
  } catch {
    return {
      detail: rawText
    };
  }
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [showMetricsModal, setShowMetricsModal] = useState(false);
  const [activeMetricTab, setActiveMetricTab] = useState("table");
  const [activeGraph, setActiveGraph] = useState("/metrics/01_confusion_matrix.png");
  const [showCloudModal, setShowCloudModal] = useState(false);
  const [activeCloudTab, setActiveCloudTab] = useState("workflow");
  const [s3Status, setS3Status] = useState(null);
  const [heatmapOpacity, setHeatmapOpacity] = useState(65);
  const [chatMessages, setChatMessages] = useState([
    { role: "assistant", text: DEFAULT_CHAT_MESSAGE }
  ]);
  const [chatInput, setChatInput] = useState("");

  useEffect(() => {
    fetch(`${apiBaseUrl}/api/s3/status`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setS3Status(data);
      })
      .catch(() => {});
  }, []);

  function resetScan() {
    setFile(null);
    setPreview("");
    setResult(null);
    setError(null);
  }

  useEffect(() => {
    if (!file) {
      setPreview("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const finalDecision = result?.summary?.finalDecision;
  const visibleModels = result?.models || [];
  const heatmapModels = visibleModels.filter((model) => model.heatmap);
  const summarySeverity = finalDecision?.severity || result?.summary?.severity || "";
  const summaryResultLabel =
    finalDecision?.resultLabel || formatPrediction(result?.summary?.finalPrediction);
  const summaryConfidence =
    finalDecision?.confidenceStatus || result?.summary?.confidenceStatus || "Waiting for analysis";
  const summaryReason =
    finalDecision?.reason || "Upload and analyze a chest X-ray to see the screening explanation.";

  useEffect(() => {
    if (!result) {
      setChatMessages([{ role: "assistant", text: DEFAULT_CHAT_MESSAGE }]);
      return;
    }

    setChatMessages([
      {
        role: "assistant",
        text: `The screening report for ${result.image.filename} is ready. I can explain the result, confidence, heatmaps, common X-ray terms, and general information about COVID-19, pneumonia, tuberculosis, and normal results.`
      }
    ]);
  }, [result]);

  function chooseFile(selectedFile) {
    // Image upload validation before sending anything to the backend.
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please choose a chest X-ray image file.");
      return;
    }

    setError(null);
    setResult(null);
    setFile(selectedFile);
  }

  async function analyze() {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Send the uploaded image to FastAPI for model prediction.
      const response = await fetch(`${apiBaseUrl}/api/predict`, {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const payload = await readErrorPayload(response);
        setError(normalizeApiError(payload));
        return;
      }

      setResult(await response.json());
    } catch (err) {
      setError({
        title: "Analysis failed",
        message: err.message || "Something went wrong while analyzing the X-ray.",
        checks: []
      });
    } finally {
      setLoading(false);
    }
  }

  function sendChatMessage(text, actionId, displayText = text) {
    const trimmed = text.trim();
    if (!trimmed) return;

    const reply = buildAssistantResponse(trimmed, actionId, result, visibleModels, file);
    setChatMessages((messages) => [
      ...messages,
      { role: "user", text: displayText.trim() || trimmed },
      { role: "assistant", text: reply }
    ]);
    setChatInput("");
  }

  function handleChatSubmit(event) {
    event.preventDefault();
    sendChatMessage(chatInput);
  }

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Chest X-ray screening</p>
          <h1>Chest X-Ray Disease Detection</h1>
        </div>
        <div className="topbar-actions">
          <span className="live-status-pill">
            <span className="live-dot" /> 4 Models Active
          </span>
          <button
            type="button"
            className="action-pill-btn cloud-pill-btn"
            onClick={() => setShowCloudModal(true)}
            title="View AWS S3 and Floci Emulator Workflow"
          >
            ☁️ AWS S3 & Floci Cloud
          </button>
          <button
            type="button"
            className="action-pill-btn"
            onClick={() => setShowMetricsModal(true)}
          >
            📊 Model Metrics & Evaluation
          </button>
          {result && (
            <button
              type="button"
              className="action-pill-btn print-btn"
              onClick={() => window.print()}
            >
              🖨️ Print Clinical Report
            </button>
          )}
        </div>
      </section>

      <section className="workspace">
        <aside className={`case-summary ${summarySeverity}`}>
          <div className="summary-header">
            <p className="eyebrow">Case Summary</p>
            <h2>{summaryResultLabel}</h2>
          </div>

          <figure className="summary-preview">
            {preview ? (
              <img src={preview} alt="Uploaded chest X-ray preview" />
            ) : (
              <div className="summary-placeholder">Upload X-ray preview</div>
            )}
            <figcaption>{file?.name || "No image selected"}</figcaption>
          </figure>

          {file && (
            <button type="button" className="reset-btn" onClick={resetScan}>
              ✕ Clear / Upload New Scan
            </button>
          )}

          <div className="summary-details">
            <article>
              <p className="label">Confidence</p>
              <strong>{summaryConfidence}</strong>
            </article>
            <article>
              <p className="label">Explanation</p>
              <span>{summaryReason}</span>
            </article>
          </div>

          <div className="summary-disclaimer">
            <strong>AI support only</strong>
            <span>This is not a final diagnosis. Screening results should be reviewed by a qualified professional.</span>
          </div>
        </aside>

        <section className="results-panel">
          <section
            className={`upload-panel ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFile(event.dataTransfer.files[0]);
            }}
          >
            <div className="upload-copy">
              <span className="step">1</span>
              <div>
                <h2>Upload an X-ray</h2>
                <p>Choose or drop a chest X-ray image to generate a screening report.</p>
              </div>
            </div>

            <div className="upload-actions">
              <label className="file-button">
                Choose image
                <input
                  type="file"
                  accept="image/*"
                  onChange={(event) => chooseFile(event.target.files[0])}
                />
              </label>

              <button className="analyze-button" disabled={!file || loading} onClick={analyze}>
                {loading ? "Analyzing..." : "Analyze X-ray"}
              </button>
            </div>

            {loading && (
              <div className="analysis-progress-banner">
                <div className="spinner-dot" />
                <div>
                  <strong>Executing AWS Cloud Workflow...</strong>
                  <span>1. Archiving to S3 ({s3Status?.bucket || "chest-xray-images"}) → 2. Deep Learning Multi-Model Inference → 3. Grad-CAM Heatmaps</span>
                </div>
              </div>
            )}

            <p className="selected-file">
              {file ? `Selected image: ${file.name}` : "No image selected yet. The preview appears in the summary panel."}
            </p>

            {error && (
              <div className="error-message">
                <strong>{typeof error === "string" ? "Analysis stopped" : error.title}</strong>
                <span>{typeof error === "string" ? error : error.message}</span>
                {typeof error !== "string" && error.stage && <small>Stage: {error.stage}</small>}
                {typeof error !== "string" && error.checks?.length > 0 && (
                  <ul>
                    {error.checks.map((check) => (
                      <li className={check.passed ? "passed" : "failed"} key={check.name}>
                        <b>{check.passed ? "Passed" : "Check"}</b>
                        {check.name}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </section>

          <section className={`final-decision ${finalDecision?.severity || ""}`}>
            <div className="section-heading">
              <span className="step">2</span>
              <div>
                <h2>Screening Result</h2>
                <p>Review the result, confidence, and recommendation.</p>
              </div>
            </div>

            <div className="decision-grid">
              <article>
                <p className="label">Final Result</p>
                <strong>{finalDecision?.resultLabel || "Waiting for analysis"}</strong>
              </article>
              <article>
                <p className="label">Confidence</p>
                <strong>{finalDecision?.confidenceStatus || "No decision yet"}</strong>
              </article>
              <article className="decision-wide">
                <p className="label">Reason</p>
                <span>{finalDecision?.reason || "Upload and analyze a chest X-ray to calculate the screening result."}</span>
              </article>
              <article className="decision-wide">
                <p className="label">Recommendation</p>
                <span>{finalDecision?.recommendation || "Review the result details after analysis."}</span>
              </article>
            </div>

            {result?.s3 && (
              <div className={`s3-badge-card ${result.s3.stored ? "stored" : "fallback"}`}>
                <div className="s3-badge-header">
                  <span className="s3-cloud-icon">☁️</span>
                  <div className="s3-badge-title-group">
                    <strong>Image Storage: {result.s3.stored ? "Archived in AWS S3" : "Local Memory"}</strong>
                    <span className="s3-provider-pill">{result.s3.provider}</span>
                  </div>
                </div>
                {result.s3.stored && (
                  <div className="s3-meta-grid">
                    <div className="s3-meta-item">
                      <span className="s3-label">S3 Bucket</span>
                      <code>{result.s3.bucket}</code>
                    </div>
                    <div className="s3-meta-item">
                      <span className="s3-label">S3 Object Key</span>
                      <code title={result.s3.key}>{result.s3.key}</code>
                    </div>
                    <div className="s3-meta-item">
                      <span className="s3-label">Region / Endpoint</span>
                      <span>{result.s3.region} ({result.s3.endpoint})</span>
                    </div>
                    {result.s3.url && (
                      <div className="s3-meta-item s3-link-item">
                        <a href={result.s3.url} target="_blank" rel="noreferrer" className="s3-view-btn">
                          ↗ View Image in S3
                        </a>
                      </div>
                    )}
                  </div>
                )}
                {result.s3.error && (
                  <small className="s3-error-note">Notice: {result.s3.error}</small>
                )}
              </div>
            )}
          </section>

          <section className="table-section">
            <div className="section-heading">
              <span className="step">3</span>
              <div>
                <h2>Model Results</h2>
                <p>Each row shows the model output and confidence.</p>
              </div>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Task</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleModels.length > 0 ? visibleModels.map((model) => (
                    <tr key={model.id}>
                      <td>
                        <strong>{model.name}</strong>
                        <span>{model.architecture}</span>
                        {model.decisionNote && <small className="decision-note">{model.decisionNote}</small>}
                      </td>
                      <td>{model.focus}</td>
                      <td>
                        <span className={`status ${model.status}`}>{formatPrediction(model.prediction)}</span>
                        {model.signalNote && <small className="signal-note">{model.signalNote}</small>}
                      </td>
                      <td>{model.confidence}%</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="4" className="placeholder-cell">
                        Model results will appear here after analysis.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section className="visual-section">
            <div className="section-heading">
              <span className="step">4</span>
              <div>
                <h2>Image Evidence</h2>
                <p>Heatmaps show which image regions influenced each model prediction.</p>
              </div>
            </div>

            <div className="heatmap-legend" aria-label="Heatmap color legend">
              <span>
                <i className="legend-hot" aria-hidden="true" />
                Red/yellow means stronger AI attention
              </span>
              <span>
                <i className="legend-cool" aria-hidden="true" />
                Blue means weaker AI attention
              </span>
            </div>

            {preview && heatmapModels.length > 0 && (
              <div className="blend-slider-row">
                <label htmlFor="heatmap-blend">Grad-CAM Overlay Opacity: {heatmapOpacity}%</label>
                <input
                  id="heatmap-blend"
                  type="range"
                  min="0"
                  max="100"
                  value={heatmapOpacity}
                  onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
                />
              </div>
            )}

            <div className="visual-grid">
              {preview && (
                <article className="visual-card">
                  <img src={preview} alt="Original uploaded X-ray" />
                  <div className="visual-caption">
                    <strong>Original X-ray</strong>
                    <span>Uploaded image used for the screening report.</span>
                  </div>
                </article>
              )}

              {heatmapModels.map((model) => (
                <article className="visual-card" key={`${model.id}-heatmap`}>
                  <div className="heatmap-blend-card">
                    {preview && <img src={preview} alt="Base radiograph" />}
                    <img
                      src={model.heatmap}
                      alt={`${model.name} Grad-CAM heatmap`}
                      style={{ opacity: heatmapOpacity / 100 }}
                    />
                  </div>
                  <div className="visual-caption">
                    <strong>Model: {model.name}</strong>
                    <dl>
                      <div>
                        <dt>Prediction</dt>
                        <dd>{formatPrediction(model.prediction)}</dd>
                      </div>
                      <div>
                        <dt>Confidence</dt>
                        <dd>{model.confidence}%</dd>
                      </div>
                      <div>
                        <dt>Heatmap</dt>
                        <dd>Highlighted areas show image regions that influenced this model output.</dd>
                      </div>
                    </dl>
                  </div>
                </article>
              ))}

              {!preview && !result && <div className="visual-placeholder">Heatmaps will appear after analysis.</div>}
              {result && heatmapModels.length === 0 && (
                <div className="visual-placeholder">Heatmap evidence is not available for this image.</div>
              )}
            </div>
          </section>

          <section className="chat-section">
            <div className="section-heading">
              <span className="step">5</span>
              <div>
                <h2>Medical Information Assistant</h2>
                <p>Ask about the result or general educational chest X-ray information.</p>
              </div>
            </div>

            <div className="chat-panel">
              <div className="chat-messages" role="log" aria-live="polite">
                {chatMessages.map((message, index) => (
                  <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                    <span>{message.role === "assistant" ? "Assistant" : "You"}</span>
                    <p>{message.text}</p>
                  </div>
                ))}
              </div>

              <div className="quick-actions" aria-label="Quick chatbot actions">
                {QUICK_ACTIONS.map((action) => (
                  <button
                    key={action.id}
                    type="button"
                    onClick={() => sendChatMessage(action.prompt, action.id, action.label)}
                  >
                    {action.label}
                  </button>
                ))}
              </div>

              <form className="chat-input-row" onSubmit={handleChatSubmit}>
                <input
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  placeholder="Ask about the result, a disease, or an X-ray term..."
                  aria-label="Ask the medical information assistant"
                />
                <button type="submit" disabled={!chatInput.trim()}>
                  Send
                </button>
              </form>
            </div>
          </section>
        </section>
      </section>

      {showMetricsModal && (
        <div className="modal-overlay" onClick={() => setShowMetricsModal(false)}>
          <div className="modal-container" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📊 Multi-Disease Model Evaluation & Clinical Metrics</h2>
              <button
                type="button"
                className="modal-close-btn"
                onClick={() => setShowMetricsModal(false)}
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              {/* Top KPI Cards */}
              <div className="modal-kpi-grid">
                <div className="kpi-card">
                  <span className="label">Validation Accuracy</span>
                  <strong>94.74%</strong>
                  <span>Peak Epoch 7 & 8</span>
                </div>
                <div className="kpi-card">
                  <span className="label">Test Accuracy</span>
                  <strong>92.22%</strong>
                  <span>711 / 771 correct</span>
                </div>
                <div className="kpi-card">
                  <span className="label">Final Training Loss</span>
                  <strong>0.0952</strong>
                  <span>Converged from 0.957</span>
                </div>
                <div className="kpi-card">
                  <span className="label">Test Dataset Size</span>
                  <strong>771</strong>
                  <span>Independent X-rays</span>
                </div>
              </div>

              {/* Tabs */}
              <div className="modal-tabs">
                <button
                  type="button"
                  className={`modal-tab-btn ${activeMetricTab === "table" ? "active" : ""}`}
                  onClick={() => setActiveMetricTab("table")}
                >
                  Class-Wise Evaluation Metrics & Matrix
                </button>
                <button
                  type="button"
                  className={`modal-tab-btn ${activeMetricTab === "graphs" ? "active" : ""}`}
                  onClick={() => setActiveMetricTab("graphs")}
                >
                  Visual Graphs Gallery (300 DPI)
                </button>
              </div>

              {activeMetricTab === "table" ? (
                <>
                  {/* Class-wise Cards */}
                  <div className="metric-cards-container">
                    <div className="class-metric-card">
                      <h3>COVID-19 <span>106 Cases</span></h3>
                      <div className="metric-row"><span>Sensitivity / Recall:</span> <b>99.06%</b></div>
                      <div className="metric-row"><span>Precision:</span> <b>100.00%</b></div>
                      <div className="metric-row"><span>F1-Score:</span> <b>99.53%</b></div>
                      <div className="metric-row"><span>Specificity:</span> <b>100.00%</b></div>
                    </div>
                    <div className="class-metric-card">
                      <h3>Normal <span>234 Cases</span></h3>
                      <div className="metric-row"><span>Sensitivity / Recall:</span> <b>77.78%</b></div>
                      <div className="metric-row"><span>Precision:</span> <b>96.30%</b></div>
                      <div className="metric-row"><span>F1-Score:</span> <b>86.05%</b></div>
                      <div className="metric-row"><span>Specificity:</span> <b>98.70%</b></div>
                    </div>
                    <div className="class-metric-card">
                      <h3>Pneumonia <span>390 Cases</span></h3>
                      <div className="metric-row"><span>Sensitivity / Recall:</span> <b>98.21%</b></div>
                      <div className="metric-row"><span>Precision:</span> <b>88.45%</b></div>
                      <div className="metric-row"><span>F1-Score:</span> <b>93.07%</b></div>
                      <div className="metric-row"><span>Specificity:</span> <b>86.88%</b></div>
                    </div>
                    <div className="class-metric-card">
                      <h3>Tuberculosis <span>41 Cases</span></h3>
                      <div className="metric-row"><span>Sensitivity / Recall:</span> <b>100.00%</b></div>
                      <div className="metric-row"><span>Precision:</span> <b>93.18%</b></div>
                      <div className="metric-row"><span>F1-Score:</span> <b>96.47%</b></div>
                      <div className="metric-row"><span>Specificity:</span> <b>99.59%</b></div>
                    </div>
                  </div>

                  {/* Confusion Matrix Table */}
                  <div className="cm-matrix-view">
                    <h4 style={{ margin: "0 0 10px 0", color: "#14213d" }}>
                      Core Multi-Class Confusion Matrix (N = 771 Test Samples)
                    </h4>
                    <table className="cm-table">
                      <thead>
                        <tr>
                          <th>Actual \ Predicted</th>
                          <th>Pred. COVID-19</th>
                          <th>Pred. Normal</th>
                          <th>Pred. Pneumonia</th>
                          <th>Pred. Tuberculosis</th>
                          <th>Total Actual</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td><b>Actual COVID-19</b></td>
                          <td className="correct-diag">105 (99.1%)</td>
                          <td>0</td>
                          <td>0</td>
                          <td>1</td>
                          <td><b>106</b></td>
                        </tr>
                        <tr>
                          <td><b>Actual Normal</b></td>
                          <td>0</td>
                          <td className="correct-diag">182 (77.8%)</td>
                          <td>50</td>
                          <td>2</td>
                          <td><b>234</b></td>
                        </tr>
                        <tr>
                          <td><b>Actual Pneumonia</b></td>
                          <td>0</td>
                          <td>7</td>
                          <td className="correct-diag">383 (98.2%)</td>
                          <td>0</td>
                          <td><b>390</b></td>
                        </tr>
                        <tr>
                          <td><b>Actual Tuberculosis</b></td>
                          <td>0</td>
                          <td>0</td>
                          <td>0</td>
                          <td className="correct-diag">41 (100%)</td>
                          <td><b>41</b></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                /* Graphs Gallery */
                <div className="graphs-gallery">
                  <div className="graph-chips">
                    {[
                      { id: "/metrics/01_confusion_matrix.png", label: "Confusion Matrix" },
                      { id: "/metrics/02_recall_sensitivity_by_class.png", label: "Sensitivity / Recall" },
                      { id: "/metrics/03_precision_by_class.png", label: "Precision" },
                      { id: "/metrics/04_f1_score_by_class.png", label: "F1-Score" },
                      { id: "/metrics/05_specificity_by_class.png", label: "Specificity" },
                      { id: "/metrics/06_all_metrics_comparison.png", label: "All Metrics Compared" },
                      { id: "/metrics/07_training_loss_curve.png", label: "Loss Curve" },
                      { id: "/metrics/08_validation_accuracy_curve.png", label: "Validation Accuracy" },
                    ].map((chip) => (
                      <button
                        key={chip.id}
                        type="button"
                        className={`graph-chip-btn ${activeGraph === chip.id ? "active" : ""}`}
                        onClick={() => setActiveGraph(chip.id)}
                      >
                        {chip.label}
                      </button>
                    ))}
                  </div>

                  <div className="graph-preview-box">
                    <img src={activeGraph} alt="Selected Metric Visualization" />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Cloud & Floci S3 Architecture Modal */}
      {showCloudModal && (
        <div className="modal-overlay" onClick={() => setShowCloudModal(false)}>
          <div className="modal-content metrics-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <p className="eyebrow">Cloud Architecture & Deployment</p>
                <h3>AWS S3 & Floci Emulator Workflow</h3>
              </div>
              <button
                type="button"
                className="close-modal-btn"
                onClick={() => setShowCloudModal(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-tabs">
              <button
                type="button"
                className={`tab-btn ${activeCloudTab === "workflow" ? "active" : ""}`}
                onClick={() => setActiveCloudTab("workflow")}
              >
                🔄 End-to-End Workflow
              </button>
              <button
                type="button"
                className={`tab-btn ${activeCloudTab === "floci" ? "active" : ""}`}
                onClick={() => setActiveCloudTab("floci")}
              >
                ⚡ Floci Local Emulator
              </button>
              <button
                type="button"
                className={`tab-btn ${activeCloudTab === "aws" ? "active" : ""}`}
                onClick={() => setActiveCloudTab("aws")}
              >
                🚀 AWS Cloud Deployment
              </button>
            </div>

            <div className="modal-body">
              {activeCloudTab === "workflow" && (
                <div className="cloud-instructions">
                  <p>
                    This system connects the <b>React Frontend</b>, <b>AWS S3 Object Storage</b> (or local <b>Floci</b> emulator),
                    and <b>FastAPI Deep Learning Backend</b> in a cohesive, production-grade cloud workflow:
                  </p>

                  <div className="arch-diagram-flow">
                    <div className="arch-node active-stage">
                      <span className="arch-step-num">1</span>
                      <h4>React Frontend</h4>
                      <p>User uploads or drops chest X-ray image (PNG/JPG). Client runs pre-flight validation.</p>
                    </div>
                    <div className="arch-node active-stage">
                      <span className="arch-step-num">2</span>
                      <h4>AWS S3 Storage</h4>
                      <p>Image bytes are archived to S3 bucket (e.g. <code>chest-xray-images/xrays/...</code>) with unique timestamp.</p>
                    </div>
                    <div className="arch-node active-stage">
                      <span className="arch-step-num">3</span>
                      <h4>PyTorch AI Inference</h4>
                      <p>FastAPI loads image, runs DenseNet-121 + 3 ResNet-18 specialists, and generates Grad-CAM attention heatmaps.</p>
                    </div>
                    <div className="arch-node active-stage">
                      <span className="arch-step-num">4</span>
                      <h4>Frontend Results</h4>
                      <p>Frontend displays consensus diagnosis, confidence, Grad-CAM overlay, S3 object URL, and AI chatbot.</p>
                    </div>
                  </div>

                  <div className="s3-meta-grid" style={{ marginTop: "16px", background: "#f8fafc", padding: "16px", borderRadius: "8px" }}>
                    <div className="s3-meta-item">
                      <span className="s3-label">Current Provider</span>
                      <strong>{s3Status?.provider || "Detecting..."}</strong>
                    </div>
                    <div className="s3-meta-item">
                      <span className="s3-label">S3 Bucket Name</span>
                      <code>{s3Status?.bucket || "chest-xray-images"}</code>
                    </div>
                    <div className="s3-meta-item">
                      <span className="s3-label">Endpoint URL</span>
                      <span>{s3Status?.endpoint || "AWS Cloud"}</span>
                    </div>
                    <div className="s3-meta-item">
                      <span className="s3-label">Connection Status</span>
                      <span style={{ color: s3Status?.connected ? "#166534" : "#b45309", fontWeight: "bold" }}>
                        {s3Status?.connected ? "● Connected" : "○ Ready / Standby"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {activeCloudTab === "floci" && (
                <div className="cloud-instructions">
                  <p>
                    <a href="https://github.com/floci-io/floci" target="_blank" rel="noreferrer">
                      <b>Floci</b>
                    </a>{" "}
                    is a free, open-source, ultra-fast local AWS emulator (~24ms startup, ~13MB memory footprint)
                    built with Quarkus and GraalVM. It lets you test real AWS S3 APIs locally with zero cloud bills!
                  </p>

                  <h4>Option A: Run Floci with Docker (Recommended)</h4>
                  <div className="code-snippet-box">
                    docker run -d -p 4566:4566 --name floci floci/floci:latest
                  </div>

                  <h4>Option B: Run Floci with CLI</h4>
                  <div className="code-snippet-box">
                    curl -fsSL https://floci.io/install.sh | sh{"\n"}
                    floci start{"\n"}
                    eval $(floci env)
                  </div>

                  <h4>Configure App Environment for Floci:</h4>
                  <div className="code-snippet-box">
                    FLOCI_ENDPOINT_URL=http://localhost:4566{"\n"}
                    AWS_REGION=us-east-1{"\n"}
                    AWS_ACCESS_KEY_ID=test{"\n"}
                    AWS_SECRET_ACCESS_KEY=test{"\n"}
                    AWS_S3_BUCKET_NAME=chest-xray-images
                  </div>
                </div>
              )}

              {activeCloudTab === "aws" && (
                <div className="cloud-instructions">
                  <h4>Deploying to Production AWS Cloud:</h4>
                  <ol>
                    <li>
                      <b>Create S3 Bucket in AWS:</b>
                      <div className="code-snippet-box">
                        aws s3 mb s3://chest-xray-images --region us-east-1
                      </div>
                    </li>
                    <li>
                      <b>Configure AWS Credentials / IAM Role:</b>
                      <p>Attach the <code>AmazonS3FullAccess</code> (or scoped S3 put/get policy) to your EC2 IAM Role or export environment variables:</p>
                      <div className="code-snippet-box">
                        export AWS_REGION="us-east-1"{"\n"}
                        export AWS_S3_BUCKET_NAME="chest-xray-images"{"\n"}
                        export AWS_ACCESS_KEY_ID="&lt;YOUR_AWS_ACCESS_KEY&gt;"{"\n"}
                        export AWS_SECRET_ACCESS_KEY="&lt;YOUR_AWS_SECRET_KEY&gt;"{"\n"}
                        # Leave FLOCI_ENDPOINT_URL empty to connect directly to AWS Cloud S3!
                      </div>
                    </li>
                    <li>
                      <b>Launch EC2 Instance (Free Tier t2.micro or t3.medium):</b>
                      <p>Run FastAPI with systemd and Nginx reverse proxy. Follow the complete step-by-step instructions in <code>cloud.md</code>.</p>
                    </li>
                  </ol>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;

