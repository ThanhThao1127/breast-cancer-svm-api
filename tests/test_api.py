from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")

    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_rejects_missing_features():
    r = client.post(
        "/predict",
        json={
            "features": {
                "mean radius": 10.0
            }
        }
    )

    assert r.status_code == 422


def test_metadata_has_30_features():
    r = client.get("/metadata")

    assert r.status_code == 200
    assert len(r.json()["feature_names"]) == 30
def test_prediction_regression():
    sample = {
        "mean radius": 17.99,
        "mean texture": 10.38,
        "mean perimeter": 122.8,
        "mean area": 1001.0,
        "mean smoothness": 0.1184,
        "mean compactness": 0.2776,
        "mean concavity": 0.3001,
        "mean concave points": 0.1471,
        "mean symmetry": 0.2419,
        "mean fractal dimension": 0.07871,
        "radius error": 1.095,
        "texture error": 0.9053,
        "perimeter error": 8.589,
        "area error": 153.4,
        "smoothness error": 0.006399,
        "compactness error": 0.04904,
        "concavity error": 0.05373,
        "concave points error": 0.01587,
        "symmetry error": 0.03003,
        "fractal dimension error": 0.006193,
        "worst radius": 25.38,
        "worst texture": 17.33,
        "worst perimeter": 184.6,
        "worst area": 2019.0,
        "worst smoothness": 0.1622,
        "worst compactness": 0.6656,
        "worst concavity": 0.7119,
        "worst concave points": 0.2654,
        "worst symmetry": 0.4601,
        "worst fractal dimension": 0.1189
    }

    r = client.post(
        "/predict",
        json={"features": sample}
    )

    assert r.status_code == 200

    data = r.json()

    assert data["model_version"] == "1.0.0"
    assert data["predicted_class"] == 0
    assert data["predicted_label"] == "malignant"