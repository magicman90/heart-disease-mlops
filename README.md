# Heart Disease Risk Prediction — MLOps Pipeline
This repository stores my submission for Assignment 01 for MLOps subject. The goal was to take a
heart disease classifier from raw data all the way to a deployed, monitored
API — so that's what's here: data prep, EDA, two trained models tracked in
MLflow, a FastAPI service wrapped in Docker, Kubernetes manifests to run it
in a cluster, and a CI/CD pipeline that ties it all together.


## Project Layout

```
heart-disease-mlops/
├── data/
│   └── raw/heart_disease_uci_raw.csv     # dataset, saved after download
├── notebooks/
│   ├── 01_eda.ipynb                      # exploratory analysis
│   └── 02_training.ipynb                 # training + MLflow tracking, notebook version
├── src/
│   ├── download_data.py                  # pulls the dataset
│   ├── preprocessing.py                  # cleaning + the sklearn Pipeline/ColumnTransformer
│   ├── eda.py                            # same EDA as the notebook, as a script
│   ├── train.py                          # trains, tunes, logs to MLflow, saves the model
│   └── app.py                            # the FastAPI app that serves predictions
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
├── models/
│   ├── best_model.joblib                 # the winning pipeline (preprocessing + classifier together)
│   └── model_metadata.joblib
├── reports/
│   ├── figures/                          # every plot — EDA and model evaluation
│   └── model_comparison.csv
├── docker/
│   ├── docker-compose.yml
│   └── prometheus.yml
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── ingress.yaml
├── .github/workflows/ci-cd.yaml          # GitHub Actions pipeline
├── screenshots/                          # Docker / K8s / MLflow / CI screenshots go here
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting It Running Locally

```bash
# set up a clean virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install everything
pip install -r requirements.txt

# grab the dataset
python src/download_data.py
```

A quick note if you're on a Mac and this download step throws
`SSL: CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate`
— that's a known python.org installer quirk where Python doesn't come with a
configured certificate store. The script falls back to `certifi`'s bundle
automatically, which fixes it in most cases. If it still doesn't work, run
`open "/Applications/Python 3.x/Install Certificates.command"` (match
whatever version you've got installed), or just `pip install --upgrade
certifi`.

```bash
# EDA — either run the script or open the notebook, they cover the same ground
python src/eda.py
# or: jupyter notebook notebooks/01_eda.ipynb

# train both models and log everything to MLflow
python src/train.py

# browse the experiment runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then visit http://localhost:5000

# run the test suite
pytest tests/ -v

# start the API
cd src
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Swagger docs live at http://localhost:8000/docs
```

### Trying a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
    "ca": 0.0, "thal": 1.0
  }'
```

which comes back with something like:
```json
{"prediction": 0, "label": "No Disease", "confidence": 0.8146, "model_name": "logistic_regression"}
```

## Docker

```bash
# build from the project root — the Dockerfile pulls in src/ and models/
docker build -t heart-disease-api:latest .

# run it
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

# check it's alive
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'

# or bring up the whole stack (API + Prometheus + Grafana) with compose
cd docker
docker compose up --build
```

For the report, I grabbed screenshots of a successful `docker build`,
`docker run`, and a working `/predict` call — they're in `screenshots/`.

## Deploying to Kubernetes

I tested this against Minikube, but it should work the same way on Docker
Desktop's built-in Kubernetes or a managed cluster like GKE/EKS/AKS.

```bash
# Minikube example
minikube start
eval $(minikube docker-env)          # so the image builds straight into Minikube's Docker
docker build -t heart-disease-api:latest .

kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
# optional, if you want ingress instead of a LoadBalancer:
kubectl apply -f k8s/ingress.yaml

kubectl get pods
kubectl get svc heart-disease-api-service

# reach it through a Minikube tunnel or NodePort
minikube service heart-disease-api-service
```

Screenshots for this section: `kubectl get pods` showing everything Running,
`kubectl get svc`, and a curl call against the exposed endpoint.

## The CI/CD Side

`.github/workflows/ci-cd.yaml` fires on every push or PR into `main` or
`develop`, and runs four jobs in sequence:

1. **Lint** — flake8, plus a black formatting check
2. **Unit tests** — installs dependencies, downloads the dataset, trains the
   model, then runs pytest. The trained model and MLflow tracking DB get
   uploaded as workflow artifacts so you can pull them down if needed.
3. **Build the Docker image** — builds it and smoke-tests `/health` to make
   sure the container actually comes up
4. **Deploy** — only runs on `main`, and right now it's a placeholder step.
   If I wanted a fully automated deploy I'd wire in real cluster credentials
   here (a `kubeconfig` secret, for instance).

I saved a screenshot of a fully green pipeline run in `screenshots/`.

## Monitoring & Logging

Every request the API handles gets logged — both to stdout and to
`api_requests.log`. On top of that:

- `GET /metrics` returns Prometheus-formatted metrics (request counts,
  latency histograms, status code breakdowns), courtesy of
  `prometheus-fastapi-instrumentator`.
- `GET /stats` is a simpler JSON version of the same idea, useful for a
  quick manual check without needing Prometheus running.
- `docker/docker-compose.yml` also spins up Prometheus and Grafana,
  pre-configured to scrape the API. Once that's running, point Grafana at
  `http://prometheus:9090` as its data source and build a dashboard from
  there.

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.885 | 0.839 | 0.929 | 0.881 | **0.966** |
| Random Forest (tuned) | 0.869 | 0.833 | 0.893 | 0.862 | 0.947 |

Logistic Regression edged out Random Forest on ROC-AUC, so that's the model
that got saved and deployed. Full experiment history — every hyperparameter
combo, all the metrics, confusion matrices, ROC curves — is either in
`reports/model_comparison.csv` or browsable directly in the MLflow UI.

## A Few Notes on Reproducibility

- Preprocessing (imputing missing values, scaling numeric features, one-hot
  encoding categoricals) all lives inside a single `sklearn.pipeline.Pipeline`
  built by `build_preprocessor()` in `src/preprocessing.py`. It gets fit once
  during training and saved as part of `models/best_model.joblib`, so the API
  doesn't need to duplicate any of that logic — it just calls
  `.predict_proba()` on the loaded pipeline directly.
- `requirements.txt` pins major version ranges rather than exact versions, so
  the environment stays reproducible without being brittle to patch releases.
- `random_state=42` is set everywhere I split or train, so the results above
  should reproduce exactly if you rerun the pipeline.
