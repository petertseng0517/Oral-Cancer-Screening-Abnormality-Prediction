import os
import sys
import pickle
import pandas as pd

# 將專案根目錄加入模組搜尋路徑，讓 src/ 下的模組可以正常 import
sys.path.insert(0, os.path.dirname(__file__))

import config
from src import preprocess, models, evaluate


def main():
    # 確保輸出目錄存在（不存在則自動建立）
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    # ──────────────────────────────────────────
    # 步驟 1：資料前處理
    #   載入原始 CSV → 去識別化 → 日期轉換 → 欄位重命名
    #   → 類別編碼 → 缺漏值處理 → 衍生特徵計算
    #   → 訓練/測試集切割 → 標準化 → SMOTE 過採樣
    # ──────────────────────────────────────────
    print("\n=== 步驟 1：資料前處理 ===")
    X_train, X_test, y_train, y_test, feature_cols, scaler = preprocess.run_preprocessing()

    # ──────────────────────────────────────────
    # 步驟 2：建立模型
    #   依照 config.py 的超參數，初始化四個分類器物件：
    #   Random Forest、Logistic Regression、SVM、Decision Tree
    # ──────────────────────────────────────────
    print("\n=== 步驟 2：建立模型 ===")
    model_dict = models.build_models(config.RANDOM_STATE)

    # ──────────────────────────────────────────
    # 步驟 3：訓練模型
    #   用訓練集（X_train, y_train）對四個模型分別執行 fit()
    #   測試集此階段完全隔離，不參與訓練
    # ──────────────────────────────────────────
    print("\n=== 步驟 3：訓練模型 ===")
    trained = models.train_all(model_dict, X_train, y_train)

    # ──────────────────────────────────────────
    # 步驟 4：評估模型
    #   用測試集（X_test, y_test）評估每個模型的預測表現
    #   計算 Accuracy、Precision、Recall、F1-score、AUC
    # ──────────────────────────────────────────
    print("\n=== 步驟 4：評估模型 ===")
    results, summary_df = evaluate.evaluate_all(trained, X_test, y_test)

    # ──────────────────────────────────────────
    # 步驟 5：輸出視覺化圖表
    #   - Confusion Matrix：顯示預測結果的正確/錯誤分布
    #   - ROC 曲線：比較各模型在不同閾值下的表現
    #   - Feature Importance：哪些特徵對 Random Forest 預測影響最大
    #   - 異常部位分布：篩檢陽性案例出現在哪些口腔部位
    # ──────────────────────────────────────────
    print("\n=== 步驟 5：輸出圖表 ===")
    evaluate.plot_confusion_matrices(
        results, y_test,
        os.path.join(config.PROCESSED_DIR, config.FIG_CONFUSION),
    )
    evaluate.plot_roc_curves(
        results,
        os.path.join(config.PROCESSED_DIR, config.FIG_ROC),
    )
    evaluate.plot_feature_importance(
        trained["Random Forest"], feature_cols,
        os.path.join(config.PROCESSED_DIR, config.FIG_IMPORTANCE),
    )

    # 異常部位圖需要原始欄位 chCheckResultPositon，重新讀取原始資料
    raw_df = pd.read_csv(config.RAW_DATA_PATH, encoding="utf-8-sig", dtype=str)
    evaluate.plot_abnormal_positions(
        raw_df.rename(columns={"chCheckResultPositon": "check_position"}),
        os.path.join(config.PROCESSED_DIR, config.FIG_POSITION),
    )

    # ──────────────────────────────────────────
    # 步驟 6：印出與儲存模型比較摘要
    #   將四個模型的指標整理成表格，存為 CSV 供後續參考
    # ──────────────────────────────────────────
    print("\n=== 步驟 6：結果摘要 ===")
    evaluate.print_summary(summary_df)

    summary_path = os.path.join(config.PROCESSED_DIR, "model_comparison.csv")
    summary_df.to_csv(summary_path)
    print(f"\n比較表已儲存：{summary_path}")

    # ──────────────────────────────────────────
    # 步驟 7：儲存模型與前處理物件
    #   將訓練好的模型、StandardScaler、特徵欄位名稱
    #   打包成 model_bundle.pkl，供 predict.py 推論時使用
    # ──────────────────────────────────────────
    print("\n=== 步驟 7：儲存模型與前處理物件 ===")
    model_bundle = {
        "models": trained,       # 四個訓練好的模型
        "scaler": scaler,        # 標準化物件（推論時需用同一個 scaler）
        "feature_cols": feature_cols,  # 特徵欄位順序（推論時需保持一致）
    }
    bundle_path = os.path.join(config.PROCESSED_DIR, "model_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"模型已儲存：{bundle_path}")


if __name__ == "__main__":
    main()
