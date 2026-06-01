import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
)
from sklearn.ensemble import RandomForestClassifier

import config

import matplotlib.font_manager as fm

def _find_chinese_font() -> str:
    candidates = ["Heiti TC", "PingFang HK", "STHeiti", "Songti SC", "Apple LiGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "DejaVu Sans"

_CHINESE_FONT = _find_chinese_font()

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = _CHINESE_FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

COLORS = ["steelblue", "tomato", "seagreen", "mediumpurple"]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "model_name": model_name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "auc":        auc(fpr, tpr),
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "fpr":        fpr,
        "tpr":        tpr,
    }


def evaluate_all(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    results = {}
    for name, model in trained_models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)

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


def plot_roc_curves(results: dict, output_path: str) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    plt.figure(figsize=(8, 6))
    for (name, r), color in zip(results.items(), COLORS):
        plt.plot(r["fpr"], r["tpr"], color=color, lw=2,
                 label=f"{name} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve 比較")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_cols: list,
    output_path: str,
) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=True)
    importance.plot(kind="barh", color="steelblue", figsize=(8, 5))
    plt.title("Feature Importance（Random Forest）")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")
    print(f"  最重要特徵：{importance.idxmax()}（{importance.max():.4f}）")


def plot_abnormal_positions(df: pd.DataFrame, output_path: str) -> None:
    plt.rcParams["font.family"] = _CHINESE_FONT
    if "check_position" not in df.columns:
        return
    pos_series = df["check_position"].dropna().astype(str)
    pos_series = pos_series[pos_series.str.strip() != ""]
    if pos_series.empty:
        return
    counts: dict = {}
    for val in pos_series:
        for code in val.split(","):
            code = code.strip()
            if code:
                counts[code] = counts.get(code, 0) + 1
    pos_df = pd.Series(counts).sort_values(ascending=False).head(15)
    pos_df.plot(kind="bar", color="steelblue", figsize=(10, 5))
    plt.title("篩檢異常位置分布（前15名）")
    plt.ylabel("出現次數")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[plot] 已儲存：{output_path}")


def print_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("模型比較表")
    print("=" * 60)
    print(summary_df.to_string())
    best = summary_df["F1-score"].astype(float).idxmax()
    print(f"\n最佳模型（F1）：{best}")
    print("=" * 60)
