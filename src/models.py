import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config


def build_models(random_state: int) -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            **config.RF_PARAMS, random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            **config.LR_PARAMS, random_state=random_state
        ),
        "SVM": SVC(
            **config.SVM_PARAMS, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            **config.DT_PARAMS, random_state=random_state
        ),
    }


def train_all(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    trained = {}
    for name, model in models.items():
        print(f"[train] 訓練 {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model
        print("完成")
    return trained
