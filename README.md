# Self-Pruning Neural Network

This repository implements the Tredence AI Engineer case study as a complete, interview-ready project rather than a single training script.

It includes:

- A custom `PrunableLinear` PyTorch layer with learnable gate scores
- A feed-forward image classifier for CIFAR-10
- Training, evaluation, sparsity analysis, checkpointing, and report generation
- A FastAPI app for model introspection and image prediction
- Unit tests for the pruning logic and gradient flow

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
|-- pyproject.toml
`-- README.md
```

## Why L1 on Gates Encourages Sparsity

Each learnable weight is multiplied by a gate in `(0, 1)` produced by `sigmoid(gate_scores)`.
Adding an L1 penalty over those gates increases the loss when too many connections remain active.
Because L1 regularization adds a constant pressure toward smaller values, it is well suited for shrinking many gates close to zero while allowing only the most useful connections to stay open.

In practice:

- Low `lambda` keeps more connections active and usually preserves accuracy
- High `lambda` creates a smaller, sparser network but can hurt accuracy
- The experiment is about finding a useful trade-off, not just maximizing either metric alone

## Quick Start

### 1. Create and activate an environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the model

```powershell
python scripts/train_and_report.py --epochs 5 --batch-size 128 --lambdas 0.0001 0.001 0.01
```

For a fast laptop smoke run before a real training session:

```powershell
python scripts/train_and_report.py --epochs 1 --batch-size 64 --train-subset 2048 --test-subset 512 --lambdas 0.0001 0.001 0.01
```

Artifacts will be created inside `artifacts/`:

- `artifacts/checkpoints/*.pt`
- `artifacts/reports/results.md`
- `artifacts/reports/gate_distribution.png`
- `artifacts/reports/results.csv`

### 3. Serve the best checkpoint with FastAPI

```powershell
$env:MODEL_CHECKPOINT="artifacts/checkpoints/best_lambda_0.0010.pt"
uvicorn app.api:app --reload
```

### 4. Example API calls

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model/summary
```

## Training Notes

- Dataset: CIFAR-10 from `torchvision.datasets`
- Model: MLP over flattened `32x32x3` images
- Gate transformation: `sigmoid(gate_scores)`
- Sparsity penalty: sum of all gate values
- Sparsity threshold: gate value `< 1e-2`
- Optional subset flags let you do a quick dry run before a full experiment

## Interview Talking Points

- You implemented pruning during training rather than as a post-processing step
- `PrunableLinear` preserves differentiability because gradients flow through both weights and gate scores
- The repo is designed like a small production project: modular code, test coverage, artifacts, and an API layer
- You can explain the sparsity versus accuracy trade-off clearly using generated metrics and plots

## Suggested Submission Checklist

- Add your trained artifact table to `artifacts/reports/results.md`
- Push the repo to GitHub with your best checkpoint excluded if it is too large
- Mention both the assignment implementation and the FastAPI serving layer in your application
