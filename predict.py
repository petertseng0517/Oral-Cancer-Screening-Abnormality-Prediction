"""
口腔癌篩檢異常結果預測
使用方式：
    python predict.py
    或直接呼叫 predict_patient() 函式
"""
import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config

BUNDLE_PATH = os.path.join(config.PROCESSED_DIR, "model_bundle.pkl")
MODEL_NAME = "Random Forest"


def load_bundle() -> dict:
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(
            f"找不到模型檔案：{BUNDLE_PATH}\n請先執行 python main.py 訓練模型。"
        )
    with open(BUNDLE_PATH, "rb") as f:
        return pickle.load(f)


def predict_patient(
    age: int,
    gender: int,
    smoking: int,
    betel_nut: int,
    indigenous: int,
    oral_discomfort: int,
    screening_count: int = 0,
    prev_result: int = -1,
    years_since_last: float = 0.0,
    model_name: str = MODEL_NAME,
    threshold: float = 0.35,
) -> dict:
    """
    預測單一病患篩檢結果。

    參數：
        age            : 年齡（整數）
        gender         : 性別（1=男, 0=女）
        smoking        : 吸菸強度（0=無, 1=已戒, 2–5=現在吸，依強度）
        betel_nut      : 嚼檳榔強度（0=無, 1=已戒, 2–5=現在嚼，依強度）
        indigenous     : 是否原住民（1=是, 0=否）
        oral_discomfort: 自覺口腔不適（1=有, 0=無）
        screening_count: 累計篩檢次數（第一次填 0）
        prev_result    : 上次篩檢結果（第一次填 -1，正常填 0，異常填 1）
        years_since_last: 距上次篩檢年數（第一次填 0）
        model_name     : 使用哪個模型（預設 Random Forest）

    回傳：
        dict，包含 prediction（0=正常, 1=異常）與 probability（異常機率）
    """
    # model_bundle.pkl 是 main.py 訓練完成後打包的成果：
    #   四個訓練好的模型、StandardScaler、特徵欄位順序，三者必須配套使用
    bundle = load_bundle()
    model = bundle["models"][model_name]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    # 將輸入的 9 個特徵組成單筆資料（順序須與訓練時的 feature_cols 一致）
    data = {
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "betel_nut": betel_nut,
        "indigenous": indigenous,
        "oral_discomfort": oral_discomfort,
        "screening_count": screening_count,
        "prev_result": prev_result,
        "years_since_last": years_since_last,
    }

    # 依 feature_cols 重新排序欄位，再套用「訓練時 fit 好的同一個」scaler 做標準化
    # 必須沿用同一個 scaler，否則新資料的數值尺度會跟模型訓練時不一致，預測就會失準
    X = pd.DataFrame([data])[feature_cols].astype(float)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    # predict_proba 回傳 [P(正常), P(異常)]，取 index=1 即為「異常機率」
    prob = model.predict_proba(X_scaled)[0][1]
    # 與門檻比較決定最終分類：機率 >= threshold 才判定為異常
    # 門檻採 0.35（低於模型預設的 0.5），刻意調降以提高 Recall
    #   → 醫療情境下「寧可多通知、不要漏掉」的權衡設計
    pred = 1 if prob >= threshold else 0

    return {
        "prediction": pred,
        "result_label": "異常（建議轉介）" if pred == 1 else "正常",
        "abnormal_probability": round(float(prob), 4),
        "threshold": threshold,
        "model_used": model_name,
    }


def interactive_predict():
    print("=" * 50)
    print("口腔癌篩檢異常結果預測系統")
    print("=" * 50)

    try:
        age            = int(input("年齡："))
        gender         = int(input("性別（1=男, 0=女）："))
        smoking        = int(input("吸菸強度（0=無, 1=已戒, 2-5=現在吸）："))
        betel_nut      = int(input("嚼檳榔強度（0=無, 1=已戒, 2-5=現在嚼）："))
        indigenous     = int(input("是否原住民（1=是, 0=否）："))
        oral_discomfort= int(input("自覺口腔不適（1=有, 0=無）："))
        screening_count= int(input("累計篩檢次數（第一次填 0）："))
        prev_result    = int(input("上次篩檢結果（第一次填 -1, 正常=0, 異常=1）："))
        years_since_last = float(input("距上次篩檢幾年（第一次填 0）："))
    except ValueError:
        print("輸入格式錯誤，請輸入數字。")
        return

    result = predict_patient(
        age=age,
        gender=gender,
        smoking=smoking,
        betel_nut=betel_nut,
        indigenous=indigenous,
        oral_discomfort=oral_discomfort,
        screening_count=screening_count,
        prev_result=prev_result,
        years_since_last=years_since_last,
    )

    print("\n" + "=" * 50)
    print(f"預測結果：{result['result_label']}")
    print(f"異常機率：{result['abnormal_probability'] * 100:.1f}%")
    print(f"使用模型：{result['model_used']}")
    print("=" * 50)


if __name__ == "__main__":
    interactive_predict()
