from fastapi.testclient import TestClient

from seedgraph import __version__
from seedgraph.api.app import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == __version__
    assert "sqlite" in data
