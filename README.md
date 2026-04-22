# Self-Pruning Neural Network

An interview-ready implementation of the Tredence AI Engineer case study: a feed-forward neural network that learns to prune its own connections during training.

Instead of pruning after training, this project attaches a learnable gate to every weight, trains those gates jointly with the classifier, and applies an L1 sparsity objective to encourage unnecessary connections to shrink toward zero.

## Overview

This repository includes:

- A custom `PrunableLinear` layer built in PyTorch
- A self-pruning MLP classifier for CIFAR-10
- End-to-end training, evaluation, checkpointing, and report generation
- Sparsity analysis across multiple `lambda` values
- A FastAPI app for prediction and model introspection
- Unit tests for gradient flow and pruning logic

## Core Idea

Each weight in the network is paired with a learnable gate score.

During the forward pass:

1. `gate_scores` are passed through a sigmoid to produce values in `(0, 1)`
2. Those gate values are multiplied element-wise with the weight matrix
3. The resulting gated weights are used in the linear transformation

This makes the model differentiable end to end while allowing it to suppress weak connections during training.

The total loss is:

```text
Total Loss = Classification Loss + λ × Sparsity Loss
```

Where:

- `Classification Loss` is cross-entropy
- `Sparsity Loss` is the sum of all gate values
- `λ` controls the trade-off between accuracy and sparsity

## Why L1 on Gates Encourages Sparsity

The L1-style penalty adds a direct cost to keeping many gates active. Since every open connection increases the sparsity term, the optimizer is encouraged to reduce non-essential gates while preserving the most useful ones for classification.

In practice:

- Lower `lambda` values usually preserve accuracy better
- Higher `lambda` values create a sparser model
- The goal is to study the accuracy-sparsity trade-off, not optimize only one side

## Project Structure

```text
self-pruning-network/
|-- app/
|   `-- api.py
|-- scripts/
|   `-- train_and_report.py
|-- src/self_pruning_network/
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   |-- reporting.py
|   |-- train.py
|   `-- utils.py
|-- tests/
|   `-- test_prunable_layer.py
|-- app.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Key Components

### `PrunableLinear`

Implemented in [src/self_pruning_network/model.py](C:/Users/sharn/OneDrive/Desktop/Self pruning Network/src/self_pruning_network/model.py).

Responsibilities:

- stores standard `weight` and `bias` parameters
- stores a learnable `gate_scores` tensor with the same shape as `weight`
- produces `gates = sigmoid(gate_scores)`
- computes `pruned_weight = weight * gates`
- performs the linear projection using the gated weights

### Training Pipeline

Implemented in [src/self_pruning_network/train.py](C:/Users/sharn/OneDrive/Desktop/Self pruning Network/src/self_pruning_network/train.py).

Responsibilities:

- trains the model on CIFAR-10
- evaluates validation and test performance
- compares multiple `lambda` values
- saves the best checkpoint for each run
- generates a report and gate distribution plot

### API Layer

Implemented in [app/api.py](C:/Users/sharn/OneDrive/Desktop/Self pruning Network/app/api.py).

Endpoints:

- `GET /` interactive upload UI
- `GET /health` service health check
- `GET /model/summary` pruning and gate statistics
- `POST /predict` image prediction endpoint
- `GET /docs` Swagger UI

## Getting Started

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the model

Standard run:

```powershell
python scripts\train_and_report.py --epochs 5 --batch-size 128 --lambdas 0.0001 0.001 0.01
```

Quick smoke run:

```powershell
python scripts\train_and_report.py --epochs 1 --batch-size 64 --train-subset 2048 --test-subset 512 --lambdas 0.0001 0.001 0.01
```

Artifacts are written to `artifacts/`:

- `artifacts/checkpoints/*.pt`
- `artifacts/reports/results.md`
- `artifacts/reports/results.csv`
- `artifacts/reports/gate_distribution.png`
- `artifacts/reports/summary.json`

### 3. Serve a trained checkpoint

```powershell
$env:MODEL_CHECKPOINT="artifacts\checkpoints\best_lambda_0.0010.pt"
python -m uvicorn app.api:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

### 4. Run tests

```powershell
pytest -q
```

## Example API Usage

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Model summary:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model/summary
```

Prediction:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -Form @{ file = Get-Item "C:\path\to\image.jpg" }
```

## Training Notes

- Dataset: CIFAR-10 via `torchvision.datasets`
- Input format: flattened `32 x 32 x 3`
- Architecture: self-pruning MLP
- Gate transform: `sigmoid(gate_scores)`
- Sparsity metric: percentage of gates below `1e-2`
- Evaluation strategy: compare multiple `lambda` values to study sparsity vs. accuracy

## Results

After training, the project generates:

- a Markdown report summarizing each `lambda`
- a CSV table for quick comparison
- a histogram of final gate values for the best-performing model

This makes it easy to explain:

- how pruning pressure changes model behavior
- how sparsity affects classification accuracy
- which `lambda` offers the best trade-off

## Interview Talking Points

- Pruning is learned during training rather than applied afterward
- The gating mechanism stays differentiable, so weights and gates can be optimized jointly
- The project is organized like a small production ML system instead of a single notebook or script
- The API layer demonstrates practical AI engineering beyond model training alone
- The report and plots make the sparsity-accuracy trade-off easy to communicate

## Limitations

- The current model is a feed-forward MLP, so image accuracy is lower than a CNN-based approach
- The API is suitable for local/demo inference, but the PyTorch dependency stack is too large for lightweight serverless deployment targets such as Vercel Functions
- Predictions on arbitrary real-world images may be uncertain because CIFAR-10 images are low resolution and domain-specific

## Submission Checklist

- Train at least three `lambda` settings
- Review `artifacts/reports/results.md`
- Keep large checkpoints out of GitHub if they exceed repository-friendly size
- Include the GitHub repo link in your application
- Be ready to explain the gating mechanism, sparsity loss, and trade-off analysis

## License

This project was built as a case-study submission and learning project. Add a license if you plan to reuse or distribute it publicly.
