import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 不開視窗，直接存檔（伺服器/Colab 環境適用）
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)
from sklearn.ensemble import RandomForestClassifier

import config

import matplotlib.font_manager as fm

# ──────────────────────────────────────────────────────────
# 字型設定：偵測系統是否有中文字型，有則使用，否則退回預設英文字型
# ──────────────────────────────────────────────────────────

def _find_chinese_font() -> str:
    candidates = ["Heiti TC", "PingFang HK", "STHeiti", "Songti SC", "Apple LiGothic",
                  "Noto Sans CJK TC", "Noto Sans TC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"

_CHINESE_FONT = _find_chinese_font()

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = _CHINESE_FONT
plt.rcParams["axes.unicode_minus"] = False  # 避免負號顯示為方塊
plt.rcParams["figure.dpi"] = 120

COLORS = ["steelblue", "tomato", "seagreen", "mediumpurple"]


# ──────────────────────────────────────────────────────────
# 單一模型評估：計算各項指標
# ──────────────────────────────────────────────────────────

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    # predict()：依照模型學到的決策邊界，對測試集每筆資料給出 0/1 預測
    y_pred = model.predict(X_test)
    # predict_proba()：回傳每筆資料屬於各類別的機率，取 index=1（異常）的機率
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC 曲線所需的 False Positive Rate 與 True Positive Rate
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "model_name": model_name,
        # Accuracy：所有預測中答對的比例（正常+異常都算）
        "accuracy":   accuracy_score(y_test, y_pred),
        # Precision：預測為「異常」的案例中，真的是異常的比例（減少誤報）
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        # Recall（Sensitivity）：實際為「異常」的案例中，被正確找出的比例（減少漏報）
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        # F1-score：Precision 與 Recall 的調和平均，兩者皆重要時的綜合指標
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        # AUC（Area Under ROC Curve）：模型區分兩類別能力的整體評估，1.0 為完美
        "auc":        auc(fpr, tpr),
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "fpr":        fpr,
        "tpr":        tpr,
    }


# ──────────────────────────────────────────────────────────
# 對所有模型逐一評估，整理成比較表
# ──────────────────────────────────────────────────────────

def evaluate_all(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

    # 將各模型指標整理成 DataFrame，每列為一個模型，方便橫向比較
    summary_df = pd.DataFrame(
        {
            name: {
                "Accuracy":  f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall":    f"{r['recall']:.4f}",
                "F1-score":  f"{r['f1']:.4f}",
                "AUC":       f"{r['auc']:.4f}",
            }
            for name, r in results.items()
        }
    ).T
    return results, summary_df


# ──────────────────────────────────────────────────────────
# 視覺化一：Confusion Matrix（混淆矩陣）
#   橫軸 = 模型預測值，縱軸 = 真實標籤
#   左上：True Negative（正常預測正確）
#   右下：True Positive（異常預測正確）
#   右上：False Positive（正常被誤判為異常）
#   左下：False Negative（異常被漏判為正常，醫療上最危險）
# ──────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict, y_test: pd.Series, output_path: str) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
            xticklabels=["正常", "異常"], yticklabels=["正常", "異常"],
        )
        axes[i].set_title(
            f"{name}\nAcc={r['accuracy']:.3f}  F1={r['f1']:.3f}  AUC={r['auc']:.3f}",
            fontsize=11,
        )
        axes[i].set_ylabel("實際")
        axes[i].set_xlabel("預測")
    plt.suptitle("Confusion Matrix 比較", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")


# ──────────────────────────────────────────────────────────
# 視覺化二：ROC 曲線
#   X軸 = FPR（誤報率），Y軸 = TPR（召回率）
#   曲線越靠近左上角、AUC 越接近 1.0，模型越好
#   對角虛線代表隨機猜測（AUC=0.5）的基準線
# ──────────────────────────────────────────────────────────

def plot_roc_curves(results: dict, output_path: str) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    plt.figure(figsize=(8, 6))
    for (name, r), color in zip(results.items(), COLORS):
        plt.plot(r["fpr"], r["tpr"], color=color, lw=2,
                 label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")  # 隨機基準線
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve 比較")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")


# ──────────────────────────────────────────────────────────
# 視覺化三：Feature Importance（Random Forest 特徵重要性）
#   Random Forest 在每棵樹的每個節點分裂時，記錄每個特徵
#   對「降低不純度（Gini impurity）」的貢獻，加總平均後即為重要性
#   數值越高，表示該特徵對預測異常越關鍵
# ──────────────────────────────────────────────────────────

def plot_feature_importance(
    model: RandomForestClassifier,
    feature_cols: list,
    output_path: str,
) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)  # 由小到大排列，最重要的在最上方
    importance.plot(kind="barh", color="steelblue", figsize=(8, 5))
    plt.title("Feature Importance（Random Forest）")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")
    print(f"  最重要特徵：{importance.idxmax()}（{importance.max():.4f}）")


# ──────────────────────────────────────────────────────────
# 視覺化四：篩檢異常部位分布
#   統計篩檢陽性案例的異常發現位置，了解哪些部位最常出現病變
# ──────────────────────────────────────────────────────────

def plot_abnormal_positions(df: pd.DataFrame, output_path: str) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    if "check_position" not in df.columns:
        return
    pos_series = df["check_position"].dropna().astype(str)
    pos_series = pos_series[pos_series.str.strip() != ""]
    if pos_series.empty:
        return
    # 部位欄位可能記錄多個部位（以逗號分隔），逐一拆分計數
    counts: dict = {}
    for val in pos_series:
        for code in val.split(","):
            code = code.strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    pos_df = pd.Series(counts).sort_values(ascending=False).head(15)  # 取前15名
    pos_df.plot(kind="bar", color="steelblue", figsize=(10, 5))
    plt.title("篩檢異常位置分布（前15名）")
    plt.ylabel("出現次數")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")


# ──────────────────────────────────────────────────────────
# 印出模型比較摘要表，並標示 F1-score 最高的模型
# ──────────────────────────────────────────────────────────

def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("模型比較表")
    print("=" * 60)
    print(summary_df.to_string())
    # 以 F1-score 為主要依據（兼顧 Precision 與 Recall）找出最佳模型
    best = summary_df["F1-score"].astype(float).idxmax()
    print(f"\n最佳模型（F1）：{best}")
    print("=" * 60)
