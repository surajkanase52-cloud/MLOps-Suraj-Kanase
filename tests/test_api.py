from fastapi.testclient import TestClient
from app.main import app


def test_health_without_lifespan():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
