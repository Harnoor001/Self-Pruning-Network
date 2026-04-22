from __future__ import annotations

import io
import os
import sys
from base64 import b64encode
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from self_pruning_network.model import GateSummary, SelfPruningMLP


app = FastAPI(title="Self-Pruning Network API", version="0.1.0")

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

_HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Self-Pruning Network Demo</title>
  <style>
    :root {
      --bg: #f7f4ec;
      --panel: rgba(255, 255, 255, 0.8);
      --text: #1e293b;
      --muted: #5b6474;
      --accent: #c96f3b;
      --accent-dark: #8f4b26;
      --border: rgba(30, 41, 59, 0.12);
      --shadow: 0 18px 45px rgba(71, 85, 105, 0.14);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(201, 111, 59, 0.18), transparent 28%),
        radial-gradient(circle at bottom right, rgba(35, 112, 137, 0.12), transparent 22%),
        linear-gradient(160deg, #f9f4ea 0%, #eef4f8 100%);
      min-height: 100vh;
    }

    .shell {
      max-width: 1080px;
      margin: 0 auto;
      padding: 40px 20px 56px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 24px;
      align-items: stretch;
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .intro {
      padding: 30px;
    }

    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }

    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(201, 111, 59, 0.1);
      color: var(--accent-dark);
      font-size: 0.92rem;
      margin-bottom: 18px;
    }

    .lead {
      font-size: 1.05rem;
      line-height: 1.7;
      color: var(--muted);
      margin: 0 0 20px;
    }

    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding: 0;
      list-style: none;
      margin: 0;
    }

    .meta li {
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(30, 41, 59, 0.06);
      color: var(--text);
      font-size: 0.92rem;
    }

    .upload {
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      justify-content: center;
    }

    .dropzone {
      border: 2px dashed rgba(201, 111, 59, 0.35);
      border-radius: 18px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.55);
    }

    .dropzone p {
      margin: 0 0 10px;
      color: var(--muted);
    }

    input[type="file"] {
      width: 100%;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: white;
    }

    button {
      appearance: none;
      border: none;
      border-radius: 14px;
      padding: 14px 18px;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      box-shadow: 0 12px 24px rgba(201, 111, 59, 0.28);
    }

    button:hover { filter: brightness(1.03); }

    .results {
      margin-top: 28px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }

    .preview, .analysis {
      padding: 24px;
    }

    h2 {
      margin: 0 0 16px;
      font-size: 1.25rem;
    }

    .preview img {
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      border-radius: 18px;
      background: #fff;
      border: 1px solid var(--border);
    }

    .empty {
      min-height: 260px;
      border-radius: 18px;
      border: 1px dashed var(--border);
      display: grid;
      place-items: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.45);
      text-align: center;
      padding: 20px;
    }

    .prediction {
      display: grid;
      gap: 12px;
    }

    .headline {
      padding: 16px;
      border-radius: 18px;
      background: rgba(35, 112, 137, 0.08);
      border: 1px solid rgba(35, 112, 137, 0.15);
    }

    .headline strong {
      display: block;
      font-size: 1.4rem;
      margin-top: 6px;
    }

    .probs {
      display: grid;
      gap: 10px;
    }

    .prob-row {
      display: grid;
      gap: 6px;
    }

    .prob-meta {
      display: flex;
      justify-content: space-between;
      font-size: 0.95rem;
    }

    .bar {
      height: 10px;
      border-radius: 999px;
      background: rgba(30, 41, 59, 0.08);
      overflow: hidden;
    }

    .bar span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(135deg, #237089, #59a0a8);
    }

    .foot {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.95rem;
      line-height: 1.6;
    }

    .links {
      margin-top: 20px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }

    .links a {
      color: var(--accent-dark);
      text-decoration: none;
      font-weight: 600;
    }

    .error {
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(185, 28, 28, 0.08);
      color: #991b1b;
      border: 1px solid rgba(185, 28, 28, 0.18);
    }

    @media (max-width: 860px) {
      .hero, .results { grid-template-columns: 1fr; }
      .shell { padding: 24px 14px 36px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="card intro">
        <div class="eyebrow">Self-Pruning Neural Network Demo</div>
        <h1>Upload an image and inspect what the model thinks.</h1>
        <p class="lead">
          This interface uses the trained checkpoint behind your FastAPI service. It resizes the uploaded image,
          runs inference, and shows both the preview and the prediction probabilities.
        </p>
        <ul class="meta">
          <li>CIFAR-10 classes</li>
          <li>Prunable MLP</li>
          <li>Soft gates with sparsity loss</li>
        </ul>
        <div class="links">
          <a href="/docs">Open API Docs</a>
          <a href="/health">Health Check</a>
          <a href="/model/summary">Model Summary</a>
        </div>
      </div>
      <form class="card upload" action="/" method="post" enctype="multipart/form-data">
        <div class="dropzone">
          <p>Select a photo for prediction. Best results come from a single centered object from a CIFAR-10 class.</p>
          <input type="file" name="file" accept="image/*" required />
        </div>
        <button type="submit">Run Prediction</button>
      </form>
    </section>

    {body}
  </div>
</body>
</html>
"""


def _render_home(
    *,
    image_data_uri: str | None = None,
    filename: str | None = None,
    prediction: dict[str, object] | None = None,
    error_message: str | None = None,
) -> HTMLResponse:
    body = ""
    if image_data_uri or prediction or error_message:
        preview_html = (
            f'<img src="{image_data_uri}" alt="Uploaded preview" />'
            if image_data_uri
            else '<div class="empty">No image preview available.</div>'
        )

        analysis_html = ""
        if prediction:
            probability_rows = []
            for class_name, value in sorted(
                prediction["probabilities"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                probability_rows.append(
                    f"""
                    <div class="prob-row">
                      <div class="prob-meta"><span>{class_name}</span><span>{value * 100:.2f}%</span></div>
                      <div class="bar"><span style="width: {value * 100:.2f}%"></span></div>
                    </div>
                    """
                )
            analysis_html = f"""
              <div class="prediction">
                <div class="headline">
                  Prediction
                  <strong>{prediction["predicted_class"]}</strong>
                  Confidence: {prediction["confidence"] * 100:.2f}%
                </div>
                <div class="probs">
                  {''.join(probability_rows)}
                </div>
                <p class="foot">
                  Uploaded file: <strong>{filename or 'unknown'}</strong><br />
                  This model was trained on CIFAR-10, so best results come from simple images matching those 10 classes.
                </p>
              </div>
            """
        elif error_message:
            analysis_html = f'<div class="error">{error_message}</div>'
        else:
            analysis_html = '<div class="empty">Prediction details will appear here after upload.</div>'

        body = f"""
        <section class="results">
          <div class="card preview">
            <h2>Uploaded Image</h2>
            {preview_html}
          </div>
          <div class="card analysis">
            <h2>Prediction Output</h2>
            {analysis_html}
          </div>
        </section>
        """

    return HTMLResponse(_HOME_TEMPLATE.format(body=body))


def _load_checkpoint_model() -> SelfPruningMLP:
    checkpoint_path = os.getenv("MODEL_CHECKPOINT")
    if not checkpoint_path:
        raise RuntimeError("MODEL_CHECKPOINT is not set.")

    path = Path(checkpoint_path)
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    config = payload["model_config"]
    model = SelfPruningMLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.3),
        use_batchnorm=config.get("use_batchnorm", True),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _predict_from_bytes(raw: bytes) -> tuple[dict[str, object], str]:
    try:
        model = _load_checkpoint_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload.") from exc

    tensor = _TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(torch.argmax(probabilities).item())

    return (
        {
            "predicted_class": _CLASS_NAMES[predicted_idx],
            "confidence": float(probabilities[predicted_idx].item()),
            "probabilities": {
                name: float(prob.item()) for name, prob in zip(_CLASS_NAMES, probabilities)
            },
        },
        image.get_format_mimetype() or "image/png",
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return _render_home()


@app.post("/", response_class=HTMLResponse)
async def home_predict(file: UploadFile = File(...)) -> HTMLResponse:
    raw = await file.read()
    try:
        prediction, mime_type = _predict_from_bytes(raw)
    except HTTPException as exc:
        return _render_home(error_message=str(exc.detail))

    image_data_uri = f"data:{mime_type};base64,{b64encode(raw).decode('utf-8')}"
    return _render_home(
        image_data_uri=image_data_uri,
        filename=file.filename,
        prediction=prediction,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model/summary")
def model_summary() -> dict[str, object]:
    try:
        model = _load_checkpoint_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary: GateSummary = model.gate_summary()
    return {
        "layers": summary.layers,
        "mean_gate_value": summary.mean_gate_value,
        "sparsity_percent": summary.sparsity_percent,
        "total_weights": summary.total_weights,
        "pruned_weights": summary.pruned_weights,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    raw = await file.read()
    prediction, _ = _predict_from_bytes(raw)
    return prediction
