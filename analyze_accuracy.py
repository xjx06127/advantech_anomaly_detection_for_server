import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
)

# =============================================================================
# 1. 시뮬레이션 설정 (AI 및 휴리스틱 동작 모사)
# =============================================================================


def mock_isolation_forest(row):
    """
    [1단계 AI 모사]
    학습된 정상 범위(체온 33~35, 심박 50~100)를 벗어나면 이상치(-1)로 판단
    """
    bt = row["body_temp"]
    hr = row["heart_rate"]

    # 학습된 정상 범위 (가정)
    if 33.0 <= bt <= 35.0 and 50 <= hr <= 100:
        return 1  # AI: 정상
    else:
        return -1  # AI: 이상치


def apply_heuristics(row):
    """
    [2단계 휴리스틱 모사]
    AI가 이상치(-1)로 판단한 것 중, 환경 적응 등을 고려하여 최종 판단
    """
    bt = row["body_temp"]
    hr = row["heart_rate"]
    et = row["env_temp"]
    eh = row["env_humidity"]

    # 1. Safe Adaptation (오경보 방지 로직)
    # 고온 환경(30도 이상)에서 체온이 약간 높으나(35.5이하), 심박은 안정적(100이하)
    if et >= 30.0 and bt <= 35.5 and hr <= 100:
        return 0  # 정상 (Safe Adaptation)

    # 2. 그 외에는 모두 위험으로 간주 (상세 질환 로직 생략, 위험=1)
    return 1  # 위험


def run_hybrid_system(df):
    """전체 데이터에 대해 AI -> 휴리스틱 파이프라인 실행"""
    ai_results = []
    hybrid_results = []

    for i, row in df.iterrows():
        # Step 1: AI Prediction
        ai_pred = mock_isolation_forest(row)

        # AI Only 결과 저장 (비교용): -1(이상) -> 1(위험), 1(정상) -> 0(정상) 변환
        ai_label = 1 if ai_pred == -1 else 0
        ai_results.append(ai_label)

        # Step 2: Hybrid Prediction
        if ai_pred == 1:
            hybrid_results.append(0)  # AI가 정상이면 최종 정상
        else:
            # AI가 이상하다고 하면 휴리스틱 체크
            final_res = apply_heuristics(row)
            hybrid_results.append(final_res)

    return ai_results, hybrid_results


# =============================================================================
# 2. 테스트 데이터셋 생성 (10개 시나리오 기반 확장)
# =============================================================================
# 앞서 정의한 10개 TC를 기반으로 노이즈를 섞어 데이터 뻥튀기 (총 200개)
np.random.seed(42)

scenarios = [
    # (개수, 환경온도, 습도, 체온, 심박, 실제라벨, 설명)
    (50, 22.0, 50, 33.8, 75, 0, "TN: Normal Office"),
    (15, 35.0, 80, 36.0, 130, 1, "TP: Heat Stroke"),
    (
        40,
        32.0,
        60,
        35.2,
        85,
        0,
        "FP-Corrected: Adaptation",
    ),  # 중요: AI는 틀리고 하이브리드는 맞아야 함
    (15, 3.0, 40, 32.5, 45, 1, "TP-H: Hypothermia"),
    (15, 25.0, 45, 33.0, 120, 1, "TP: Tachycardia"),
    (15, 28.0, 55, 35.8, 95, 1, "TP: Hyperthermia"),
    (20, 25.0, 50, 34.2, 98, 0, "TN: Borderline HeartRate"),
    (10, 5.0, 50, 32.0, 70, 1, "TP-H: Cold Env Temp drop"),
    (10, 25.0, 50, 33.0, 52, 0, "TN: Low HeartRate Normal"),
    (10, 22.0, 50, 32.5, 48, 1, "TP: Bradycardia"),
]

data = []
for n, et, eh, bt, hr, label, desc in scenarios:
    for _ in range(n):
        # 약간의 랜덤 노이즈 추가하여 데이터 생성
        row = {
            "env_temp": et + np.random.uniform(-1, 1),
            "env_humidity": eh + np.random.uniform(-5, 5),
            "body_temp": bt + np.random.uniform(-0.1, 0.2),
            "heart_rate": int(hr + np.random.randint(-2, 3)),
            "label": label,
            "desc": desc,
        }
        data.append(row)

df = pd.DataFrame(data)

# =============================================================================
# 3. 모델 실행 및 평가
# =============================================================================
y_true = df["label"]
y_pred_ai, y_pred_hybrid = run_hybrid_system(df)


# 성능 지표 계산
def print_metrics(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"[{name} Performance]")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Precision: {prec:.4f} (오경보가 적을수록 1에 가까움)")
    print(f" - Recall   : {rec:.4f} (위험을 잘 잡을수록 1에 가까움)")
    print("-" * 30)


print("\n" + "=" * 50)
print(" >> [시뮬레이션 결과] AI 단독 vs 하이브리드 비교")
print("=" * 50)
print_metrics("AI Only Model", y_true, y_pred_ai)
print_metrics("Hybrid Model ", y_true, y_pred_hybrid)

# =============================================================================
# 4. 시각화 (Confusion Matrix)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 4-1. AI Only Confusion Matrix
cm_ai = confusion_matrix(y_true, y_pred_ai)
sns.heatmap(
    cm_ai,
    annot=True,
    fmt="d",
    cmap="Reds",
    ax=axes[0],
    cbar=False,
    annot_kws={"size": 16},
)
axes[0].set_title("AI Only Model\n(High False Alarm)", fontsize=15, fontweight="bold")
axes[0].set_xlabel("Predicted", fontsize=12)
axes[0].set_ylabel("Actual", fontsize=12)
axes[0].set_xticklabels(["Normal", "Danger"])
axes[0].set_yticklabels(["Normal", "Danger"])

# 4-2. Hybrid Model Confusion Matrix
cm_hybrid = confusion_matrix(y_true, y_pred_hybrid)
sns.heatmap(
    cm_hybrid,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    cbar=False,
    annot_kws={"size": 16},
)
axes[1].set_title(
    "Hybrid Model (AI + Heuristic)\n(Reduced False Alarm)",
    fontsize=15,
    fontweight="bold",
)
axes[1].set_xlabel("Predicted", fontsize=12)
axes[1].set_ylabel("Actual", fontsize=12)
axes[1].set_xticklabels(["Normal", "Danger"])
axes[1].set_yticklabels(["Normal", "Danger"])

plt.tight_layout()
plt.savefig("reliability_comparison.png")  # 이미지 저장
plt.show()
