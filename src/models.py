import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config


def build_models(random_state: int) -> dict:
    # 依照 config.py 中設定的超參數，初始化四個分類器
    # random_state 固定隨機種子，確保每次執行結果一致（可重現）
    return {
        # 集成學習：建立多棵決策樹並取投票結果，抗雜訊能力強
        "Random Forest": RandomForestClassifier(
            **config.RF_PARAMS, random_state=random_state
        ),
        # 線性模型：最大化兩類別的決策邊界間距，結果可解釋性高
        "Logistic Regression": LogisticRegression(
            **config.LR_PARAMS, random_state=random_state
        ),
        # 核函數方法：用 RBF kernel 將資料映射到高維空間後做線性分類
        "SVM": SVC(
            **config.SVM_PARAMS, random_state=random_state
        ),
        # 樹狀模型：依特徵閾值逐層切分，結構最直觀、最易視覺化
        "Decision Tree": DecisionTreeClassifier(
            **config.DT_PARAMS, random_state=random_state
        ),
    }


def train_all(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    # 對每個模型呼叫 fit()，讓模型從訓練集學習特徵與標籤的對應關係
    # fit() 完成後，模型內部參數（如樹的分支規則、邊界係數）即固定下來
    trained = {}
    for name, model in models.items():
        print(f"[train] 訓練 {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)  # 核心訓練步驟
        trained[name] = model
        print("完成")
    return trained
