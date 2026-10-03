from pathlib import Path
import json
import joblib
import sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    roc_auc_score
)

# 2. NẠP DỮ LIỆU VÀ KIỂM TRA BAN ĐẦU

bundle = load_breast_cancer(as_frame=True)

X = bundle.data
y = bundle.target
sample = bundle.data.iloc[0].to_dict()

payload = {
    "features": {
        k: float(v)
        for k, v in sample.items()
    }
}

print(payload)

with open("sample_request.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("Sample payload saved to sample_request.json")

print(X.shape)                    # (569, 30)
print(y.value_counts())           # Phân bố 2 lớp
print(bundle.target_names)        # ['malignant', 'benign']
print(X.isna().sum().sum())       # Kiểm tra giá trị thiếu
print(X.describe().T)             # Thống kê mô tả

# 3. CHIA DỮ LIỆU 80/20

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 4. TẠO PIPELINE VÀ KHÔNG GIAN SIÊU THAM SỐ

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ))
])

param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.001, 0.01, 0.1]
}

# 5. HUẤN LUYỆN VỚI CROSS-VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="recall_macro",
    cv=cv,
    n_jobs=-1,
    refit=True
)

search.fit(X_train, y_train)

model = search.best_estimator_

print("Best parameters:", search.best_params_)

# 6. ĐÁNH GIÁ TRÊN TẬP KIỂM TRA

pred = model.predict(X_test)

proba_malignant = model.predict_proba(X_test)[:, 0]

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=["malignant", "benign"]
    )
)

print(
    "Sensitivity:",
    recall_score(
        y_test,
        pred,
        pos_label=0
    )
)

print(
    "Precision malignant:",
    precision_score(
        y_test,
        pred,
        pos_label=0
    )
)

print(
    "ROC-AUC malignant:",
    roc_auc_score(
        (y_test == 0).astype(int),
        proba_malignant
    )
)

# 7. HOÀN THIỆN VÀ LƯU MÔ HÌNH

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


# Lưu mô hình
joblib.dump(
    model,
    ARTIFACT_DIR / "breast_cancer_svm.joblib"
)


# Lưu thông tin mô hình
metadata = {
    "model_name": "breast-cancer-svm-rbf",
    "model_version": "1.0.0",
    "feature_names": list(X.columns),
    "class_mapping": {
        "0": "malignant",
        "1": "benign"
    },
    "best_params": search.best_params_,
    "sklearn_version": sklearn.__version__,
    "warning": "Educational use only; not a medical diagnosis."
}

(ARTIFACT_DIR / "metadata.json").write_text(
    json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print("\nModel and metadata saved successfully.")