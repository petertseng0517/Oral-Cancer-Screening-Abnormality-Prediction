import os
import sys
import pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import config
from src import preprocess, models, evaluate


def main():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    print("\n=== 步驟 1：資料前處理 ===")
    X_train, X_test, y_train, y_test, feature_cols, scaler = preprocess.run_preprocessing()

    print("\n=== 步驟 2：建立模型 ===")
    model_dict = models.build_models(config.RANDOM_STATE)

    print("\n=== 步驟 3：訓練模型 ===")
    trained = models.train_all(model_dict, X_train, y_train)

    print("\n=== 步驟 4：評估模型 ===")
    results, summary_df = evaluate.evaluate_all(trained, X_test, y_test)

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

    raw_df = pd.read_csv(config.RAW_DATA_PATH, encoding="utf-8-sig", dtype=str)
    evaluate.plot_abnormal_positions(
        raw_df.rename(columns={"chCheckResultPositon": "check_position"}),
        os.path.join(config.PROCESSED_DIR, config.FIG_POSITION),
    )

    print("\n=== 步驟 6：結果摘要 ===")
    evaluate.print_summary(summary_df)

    summary_path = os.path.join(config.PROCESSED_DIR, "model_comparison.csv")
    summary_df.to_csv(summary_path)
    print(f"\n比較表已儲存：{summary_path}")

    print("\n=== 步驟 7：儲存模型與前處理物件 ===")
    model_bundle = {
        "models": trained,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    bundle_path = os.path.join(config.PROCESSED_DIR, "model_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"模型已儲存：{bundle_path}")


if __name__ == "__main__":
    main()
