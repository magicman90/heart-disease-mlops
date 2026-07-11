# Heart Disease Risk Prediction — End-to-End MLOps Project

**Course:** Machine Learning Operations (MLOps) AIMLCZG523 — Assignment 01
**Dataset:** UCI Heart Disease (Cleveland) Dataset — 303 records, 13 features, binary target

A production-style ML pipeline that predicts heart disease risk from patient
health data and serves predictions via a monitored, containerized, Kubernetes-deployable
FastAPI service.

---

## 1. Project Structure

```
heart-disease-mlops/
├── data/
│   └── raw/heart_disease_uci_raw.csv     # downloaded dataset
├── notebooks/
│   ├── 01_eda.ipynb                      # exploratory data analysis
│   └── 02_training.ipynb                 # model training & MLflow tracking (notebook form)
├── src/
│   ├── download_data.py                  # dataset acquisition
│   ├── preprocessing.py                  # cleaning + sklearn Pipeline/ColumnTransformer
│   ├── eda.py                            # EDA plot generation (script form)
│   ├── train.py                          # model training, tuning, MLflow logging, packaging
│   └── app.py                            # FastAPI serving application
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── models/
│   ├── best_model.joblib                 # full pipeline (preprocessing + classifier)
│   └── model_metadata.joblib
├── reports/
│   ├── figures/                          # EDA + evaluation plots
│   └── model_comparison.csv
├── docker/
│   ├── docker-compose.yml
│   └── prometheus.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
├── .github/workflows/ci-cd.yaml          # GitHub Actions CI/CD pipeline
├── screenshots/                          # <- put your Docker/K8s/MLflow/CI screenshots here
├── Dockerfile
├── requirements.txt
└── README.md
```

## 2. Setup — Run Locally (Python / Jupyter)

```bash
# 1. Create a clean environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
python src/download_data.py
```

> **macOS troubleshooting:** if this fails with `SSL: CERTIFICATE_VERIFY_FAILED
> ... unable to get local issuer certificate`, your Python install doesn't have
> a configured certificate trust store yet (common with python.org installers).
> The script already tries `certifi`'s bundle automatically as a first fix, but
> if it still fails, run `open "/Applications/Python 3.x/Install
> Certificates.command"` (match your installed version) or
> `pip install --upgrade certifi`.

```bash
# 4. Run EDA (either the script or the notebook)
python src/eda.py
# or: jupyter notebook notebooks/01_eda.ipynb

# 5. Train models + log to MLflow
python src/train.py

# 6. Inspect experiments in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000

# 7. Run unit tests
pytest tests/ -v

# 8. Start the API
cd src
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/docs for interactive Swagger UI
```

### Sample request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0.0, "thal": 1.0
  }'
```

Response:
```json
{"prediction": 0, "label": "No Disease", "confidence": 0.8146, "model_name": "logistic_regression"}
```

## 3. Docker

```bash
# Build (run from the project root, since Dockerfile COPYs src/ and models/)
docker build -t heart-disease-api:latest .

# Run
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

# Verify
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'

# Or use docker-compose (also brings up Prometheus + Grafana)
cd docker
docker compose up --build
```

Take screenshots of: the successful `docker build`, `docker run`, and a working
`/predict` call — save them to `screenshots/`.

## 4. Kubernetes Deployment

Works with Minikube, Docker Desktop Kubernetes, or a managed cluster (GKE/EKS/AKS).

```bash
# Example: Minikube
minikube start
eval $(minikube docker-env)          # build the image directly into Minikube's Docker
docker build -t heart-disease-api:latest .

kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
# optional:
kubectl apply -f k8s/ingress.yaml

kubectl get pods
kubectl get svc heart-disease-api-service

# Access via Minikube tunnel (LoadBalancer) or NodePort:
minikube service heart-disease-api-service
```

Take screenshots of: `kubectl get pods` (Running), `kubectl get svc`, and a
successful curl against the exposed endpoint — save them to `screenshots/`.

## 5. CI/CD Pipeline

`.github/workflows/ci-cd.yaml` runs on every push/PR to `main`/`develop`:

1. **Lint** — flake8 + black formatting check
2. **Unit tests** — downloads data, trains the model, runs pytest, uploads the
   trained model + MLflow tracking DB as workflow artifacts
3. **Build Docker image** — builds the image and smoke-tests `/health`
4. **Deploy** (main branch only) — placeholder `kubectl apply` step; wire this
   to your actual cluster credentials (e.g. via `kubeconfig` secret) if you
   want a fully automated deploy

Screenshot a green pipeline run and save it to `screenshots/`.

## 6. Monitoring & Logging

- All API requests are logged to `api_requests.log` (also visible in the container's stdout).
- `GET /metrics` exposes Prometheus-format metrics (request counts, latency histograms, status codes) via `prometheus-fastapi-instrumentator`.
- `GET /stats` exposes simple JSON counters for a quick human-readable check.
- `docker/docker-compose.yml` includes optional Prometheus + Grafana services
  pre-wired to scrape the API — bring them up with `docker compose up` and
  build a Grafana dashboard against the Prometheus data source
  (`http://prometheus:9090`).

## 7. Model & Experiment Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.885 | 0.839 | 0.929 | 0.881 | **0.966** |
| Random Forest (tuned) | 0.869 | 0.833 | 0.893 | 0.862 | 0.947 |

**Selected model:** Logistic Regression (best ROC-AUC on held-out test set).
See `reports/model_comparison.csv` and the MLflow UI for full experiment history
(hyperparameters, all metrics, confusion matrices, ROC curves).

## 8. Reproducibility Notes

- All preprocessing (imputation, scaling, one-hot encoding) is encapsulated in
  a single `sklearn.pipeline.Pipeline` + `ColumnTransformer`
  (`src/preprocessing.py::build_preprocessor`), fit once during training and
  saved as part of `models/best_model.joblib`. The API never re-implements
  preprocessing logic — it just calls `.predict_proba()` on the loaded pipeline.
- `requirements.txt` pins major version ranges for a clean, repeatable environment.
- Random seeds are fixed (`random_state=42`) throughout for reproducible splits/training.
