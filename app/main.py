from pathlib import Path
import json
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. Xác định đường dẫn đến thư mục gốc của dự án
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts/breast_cancer_svm.joblib"
META_PATH = BASE_DIR / "artifacts/metadata.json"


# ============================================================
# 2. Nạp mô hình và metadata
# ============================================================

model = joblib.load(MODEL_PATH)

metadata = json.loads(
    META_PATH.read_text(encoding="utf-8")
)


# ============================================================
# 3. Khởi tạo FastAPI
# ============================================================

app = FastAPI(
    title="Breast Cancer SVM API",
    version=metadata["model_version"],
    description="Educational demonstration only"
)


# ============================================================
# 4. Mô hình dữ liệu đầu vào
# ============================================================

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Exactly 30 named numeric features"
    )


# ============================================================
# 5. Mô hình dữ liệu đầu ra
# ============================================================

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str


# ============================================================
# 6. Kiểm tra 30 đặc trưng và tạo vector đầu vào
# ============================================================

def build_vector(payload: PredictionRequest) -> np.ndarray:

    # Lấy đúng thứ tự 30 đặc trưng từ metadata
    expected = metadata["feature_names"]

    # Các trường mà người dùng gửi lên
    received = set(payload.features)

    # Tìm trường bị thiếu
    missing = sorted(
        set(expected) - received
    )

    # Tìm trường gửi dư
    extra = sorted(
        received - set(expected)
    )

    # Nếu thiếu hoặc dư trường thì báo lỗi 422
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "missing": missing,
                "extra": extra
            }
        )

    # Tạo vector theo đúng thứ tự mà mô hình yêu cầu
    values = np.array(
        [[payload.features[name] for name in expected]],
        dtype=float
    )

    # Kiểm tra NaN, vô cực,...
    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="Features must be finite numbers"
        )

    # Kiểm tra giá trị quá lớn, bất thường
    if np.abs(values).max() > 10000:
        raise HTTPException(
            status_code=422,
            detail="Features contain biologically invalid values"
        )

    return values


# ============================================================
# 7. GET /
# ============================================================

@app.get("/")
def root():
    return {
        "service": "Breast Cancer SVM API",
        "docs": "/docs",
        "warning": metadata["warning"]
    }


# ============================================================
# 8. GET /health
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": metadata["model_version"]
    }


# ============================================================
# 9. GET /metadata
# ============================================================

@app.get("/metadata")
def get_metadata():
    return metadata


# ============================================================
# 10. POST /predict
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(payload: PredictionRequest):

    # Kiểm tra dữ liệu và tạo vector
    x = build_vector(payload)

    # Dự đoán lớp
    predicted_class = int(
        model.predict(x)[0]
    )

    # Lấy xác suất của hai lớp
    probabilities = model.predict_proba(x)[0]

    # Lấy danh sách class của SVM
    classes = list(
        model.named_steps["svc"].classes_
    )

    # Ghép class với xác suất tương ứng
    p = {
        int(c): float(v)
        for c, v in zip(classes, probabilities)
    }

    # Trả kết quả
    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][
            str(predicted_class)
        ],
        probability_malignant=p[0],
        probability_benign=p[1],
        model_version=metadata["model_version"],
        warning=metadata["warning"]
    )