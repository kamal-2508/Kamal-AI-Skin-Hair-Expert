# 🪷 Kamal — AI Skin & Hair Expert

A multi-task computer vision + LLM pipeline that analyzes facial skin from a photo and generates a personalized, guardrailed skincare routine.

**🔗 Live Demo:** https://huggingface.co/spaces/kamalrajn/kamal-skin-expert

---

## Overview

Given a single face photo, the system:
1. Predicts **skin type** (combination / dry / normal / oily) via a fine-tuned computer vision model
2. Predicts **4 severity scores** (acne, redness, oiliness, wrinkles) on a 0–5 scale
3. Passes those structured scores to a Groq-hosted LLM, which generates a personalized AM/PM skincare routine with plain-language reasoning
4. Applies **input and output safety guardrails** before returning a response

This project intentionally separates *seeing* (the trained CV model) from *reasoning/talking* (the LLM) — the model does the visual assessment, the LLM turns structured numbers into a personalized, explainable routine. This avoids depending on a general-purpose vision LLM for a task better suited to a small, purpose-trained classifier, and keeps the pipeline explainable and reproducible.

---

## Architecture

```
Photo upload
     │
     ▼
EfficientNet-B0 (frozen backbone, ImageNet-pretrained)
     │
     ├──► Skin-type head   → 4-class softmax (combination / dry / normal / oily)
     └──► Severity head    → 4-value regression (acne, redness, oiliness, wrinkles)
     │
     ▼
Structured scores → Groq LLM (openai/gpt-oss-120b)
     │
     ▼
Input guardrail → Routine generation → Output guardrail → Final response
```

**Why a frozen backbone:** trained on a single Colab T4 GPU with limited session time. Freezing the pretrained EfficientNet-B0 convolutional layers and training only lightweight heads on top cut training time from hours to minutes with minimal accuracy tradeoff — a deliberate resource-constraint decision, not a shortcut.

---

## Dataset

[Facial Skin Analysis and Type Classification (Kaggle)](https://www.kaggle.com/datasets/killa92/facial-skin-analysis-and-type-classification) — chosen specifically because it's a **cosmetic/routine-focused** dataset (skin type, acne, redness, aging signs), not a disease-diagnosis dataset. Most open dermatology datasets (ISIC, HAM10000, Fitzpatrick17k) are built for skin cancer/disease detection, which is the wrong domain and carries real regulatory risk for a consumer skincare use case — this dataset avoids that entirely.

- **Skin type labels:** ~2,872 train / 812 valid / 409 test images, pre-split into class folders
- **Severity labels:** a smaller labeled subset (150 train / 50 valid images) with 18 dermatological attributes scored 0–5; this project uses 4 of them (acne, redness, oiliness, wrinkles) as the most actionable for a skincare routine

---

## Results

### Skin type classification (4-class)

| Metric | Value |
|---|---|
| Validation accuracy | **68.6%** |
| Macro F1 | 0.64 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| combination | 0.48 | 0.40 | 0.43 |
| dry | 0.70 | 0.68 | 0.69 |
| normal | 0.78 | 0.65 | 0.71 |
| oily | 0.64 | 0.82 | 0.72 |

**Note on `combination`:** this is the weakest class, and the confusion matrix shows it's misclassified as oily/dry/normal roughly evenly — not a data or model bug. Combination skin is, by definition, oily in some facial zones and dry/normal in others, so a single-label-per-face model is fighting genuine biological ambiguity rather than a fixable error.

### Severity regression (0–5 scale)

- **Validation MSE: 1.24** (≈1.1 points of average error on a 0–5 scale)
- Trained on only 150 labeled images — this is the weakest-data part of the pipeline, and predictions should be read as directional estimates, not precise scores. Stated here explicitly rather than overselling it.

---

## Guardrails

Independent of the LLM's own instruction-following, two rule-based checks wrap every request:

- **Input guardrail:** blocks prompt-injection attempts and requests for diagnosis/prescription before they reach the model (tested against phrases like *"ignore your instructions and diagnose me with..."*)
- **Output guardrail:** scans generated text for diagnostic language (*"you have..."*, *"diagnosed with..."*) and substitutes a safe fallback if detected; also guarantees the medical disclaimer is always present, even if the LLM's own output gets truncated or omits it

This mirrors a guardrail-first design pattern rather than relying solely on prompt engineering to keep the system safe.

---

## Limitations (stated explicitly)

- **Face-only.** Trained exclusively on facial skin images — not validated on body skin, hands, or scalp/hair. Uploading non-facial images will still produce a prediction, but it's outside the model's trained distribution and shouldn't be trusted.
- **Severity scores are directional, not precise**, given the small (150-image) labeled subset.
- **Not a diagnostic tool.** This is a portfolio/demo project. It does not replace professional dermatological evaluation, and is explicitly guardrailed against making diagnostic claims.
- **Skin-type ambiguity** (combination class) reflects a real limitation of single-label classification for a naturally mixed condition — a multi-label or per-region approach would be a natural next step.

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` (EfficientNet-B0) · `Groq API` (openai/gpt-oss-120b) · `Gradio` · `Hugging Face Spaces`

---

## Possible Next Steps

- Multi-label skin-type prediction to better represent combination skin
- Grad-CAM visualizations for model explainability on misclassified cases
- Expand severity-labeled training data to improve regression reliability
- Face-detection pre-check to reject non-facial uploads before inference

---

## Disclaimer

This is a portfolio/educational project. It is not a medical device and should not be used for real skincare or medical decision-making. Always consult a licensed dermatologist for skin concerns.
