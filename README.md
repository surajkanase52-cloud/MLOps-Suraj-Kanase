# Heart Disease MLOps Project

End-to-end binary classification, experiment tracking, API serving, CI, Docker, Kubernetes and monitoring.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data
python -c "from src.data import load_and_clean; load_and_clean()"
python -m src.eda
python -m src.train
mlflow ui
uvicorn app.main:app --reload
```

## Test prediction
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

## Docker
```bash
docker build -t heart-disease-api:1.0 .
docker run --rm -p 8000:8000 heart-disease-api:1.0
```

## Minikube
```bash
minikube start
minikube image load heart-disease-api:1.0
kubectl apply -f k8s/deployment.yaml
minikube service heart-api-service --url
```
