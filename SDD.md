# SDD — 口腔癌篩檢異常結果預測系統
**Software Design Document**
版本：1.3 | 日期：2026-05-07

---

## 1. 系統概述

| 項目 | 說明 |
|------|------|
| 題目 | 以機器學習預測口腔癌篩檢異常結果 |
| 問題類型 | 二元分類（Binary Classification） |
| Label | `result`：1 = 異常，0 = 正常（由 `chResult` 轉換） |
| 資料來源 | 醫院四癌篩檢歷年資料庫（口腔癌部分，去識別化後） |
| 比較模型 | Random Forest、Logistic Regression、SVM、Decision Tree |

---

## 2. 目錄結構

```
final-project/
├── SDD.md
├── CLAUDE.md
├── requirements.txt
├── config.py          # 所有可調整設定集中於此
├── main.py            # 主程式入口，串接所有步驟
├── src/
│   ├── preprocess.py  # 資料載入、清理、特徵工程
│   ├── models.py      # 模型定義與訓練
│   └── evaluate.py    # 評估指標與圖表輸出
├── data/
│   ├── raw/           # 原始資料（不進版控）
│   └── processed/     # 處理後資料與模型輸出
└── venv/              # 不進版控
```

---

## 3. 原始欄位說明

| 原始欄位 | 說明 | 值域 | 處理方式 |
|---------|------|------|---------|
| `chMRNo` | 身分證字號 | 字串 | 產生匿名 `patient_id` 後刪除 |
| `chBirthday` | 出生日期（民國年） | 7位數，如 `0680517` | 轉換後計算 `age`，再刪除 |
| `chSex` | 性別 | `1`=男, `0`=女 | 已是數值，直接使用 → `gender` |
| `chSmoking` | 吸菸強度 | 0–5（見下表） | 保留為有序數值 → `smoking` |
| `chBinglang` | 嚼檳榔強度 | 0–5（見下表） | 保留為有序數值 → `betel_nut` |
| `chAboriginal` | 是否原住民 | `Y`/`N` | Y=1, N=0 → `indigenous` |
| `chOralAwareSymptom` | 自覺口腔不適 | `0`=無, `1`=有 | 已是數值 → `oral_discomfort` |
| `chCheckResultPositon` | 篩檢異常位置 | 逗號分隔代號（見下） | **僅供 EDA，不做特徵** |
| `chScreeningDate` | 本次篩檢日期（民國年） | 7位數，如 `0680517` | 計算 `age`、`screening_year`、`years_since_last` 後刪除 |
| `chResult` | 本次篩檢結果 | 代號（見下） | 轉換為 `result`（0/1） |

### 吸菸 / 嚼檳榔強度編碼（chSmoking / chBinglang）

| 值 | 說明 |
|----|------|
| 0 | 無（從未使用） |
| 1 | 已戒 |
| 2 | 使用 10 年以下，每天少於 20 支/顆 |
| 3 | 使用 10 年以下，每天 20 支/顆及以上 |
| 4 | 使用超過 10 年，每天少於 20 支/顆 |
| 5 | 使用超過 10 年，每天 20 支/顆及以上 |

> 保留為有序數值（ordinal 0–5），值越大代表使用強度越高。

### 篩檢結果代號（chResult）→ Label

資料中 `chResult` 為**整數型別**，實際觀察到的值如下：

| 代號（整數） | 筆數 | 轉換為 result |
|------------|------|--------------|
| `0` | 16,524 | **0**（正常） |
| `73` | 2,528 | **1**（異常） |
| `76` | 348 | **1** |
| `72` | 269 | **1** |
| `11` | 258 | **1** |
| `71` | 203 | **1** |
| `10` | 166 | **1** |
| `4` | 163 | **1** |
| `8` | 157 | **1** |
| 其他非零值 | 394 | **1** |

> 編碼邏輯：`result = 0 if chResult == 0 else 1`（整數比較，非字串）

### 篩檢異常位置代號（chCheckResultPositon，僅 EDA 用）

| 代號 | 部位 |
|------|------|
| AU/AD | 口唇 |
| BR/BL | 頰黏膜 |
| CR/CL | 臼齒後三角區 |
| DR/DL | 上牙齦/齒槽黏膜 |
| ER/EL | 下牙齦/齒槽黏膜 |
| FR/FL | 舌 |
| GR/GL | 口底黏膜 |
| HR/HL | 硬腭 |
| IR/IL | 軟腭 |
| JR/JL | 扁桃體 |
| KR/KL | 口咽後壁黏膜 |
| MR/ML | 頸部腫塊 |
| L | 其他 |

---

## 4. 最終特徵清單（10 個特徵 + 1 個 Label）

### 直接欄位（5個，轉換自原始欄位）

| 標準名稱 | 來源欄位 | 類型 | 值域 |
|---------|---------|------|------|
| `gender` | `chSex` | binary int | 1=男, 0=女 |
| `smoking` | `chSmoking` | ordinal int | 0–5 |
| `betel_nut` | `chBinglang` | ordinal int | 0–5 |
| `indigenous` | `chAboriginal` | binary int | 1=是, 0=否 |
| `oral_discomfort` | `chOralAwareSymptom` | binary int | 1=有, 0=無 ⚠️ |

> ⚠️ `oral_discomfort`：複合症狀欄位，可能包含與篩檢直接相關的症狀，存在 data leakage 風險，列為研究限制。

### 程式計算欄位（5個，衍生特徵）

| 標準名稱 | 計算來源 | 計算方式 |
|---------|---------|---------|
| `age` | `chBirthday` + `chScreeningDate` | 篩檢日期 − 出生日期（年） |
| `screening_year` | `chScreeningDate` | 取西元年份 |
| `screening_count` | `patient_id` + `chScreeningDate` 排序 | 同一病患截至本筆的累計次數（從 0 起算） |
| `prev_result` | `patient_id` + `result` 排序 | 上一筆 result（第一次填 -1，代表無紀錄） |
| `years_since_last` | `patient_id` + `chScreeningDate` 排序 | 本次 − 上次篩檢日期（年，第一次填 0） |

### Label

| 標準名稱 | 來源欄位 | 編碼邏輯 |
|---------|---------|---------|
| `result` | `chResult` | `0`（整數）→ 0，其他 → 1 |

---

## 5. config.py — 設定檔

```python
# 路徑
RAW_DATA_PATH = "data/raw/oral0507-v1.csv"
PROCESSED_DIR = "data/processed/"

# 個資欄位（去識別化時刪除）
PATIENT_ID_COL = "chMRNo"
PII_COLS = ["chMRNo", "chBirthday"]  # chScreeningDate 計算後另行刪除

# 欄位對應（原始欄位名稱 → 標準名稱）
RENAME_MAP = {
    "chSex":              "gender",
    "chSmoking":          "smoking",
    "chBinglang":         "betel_nut",
    "chAboriginal":       "indigenous",
    "chOralAwareSymptom": "oral_discomfort",
    "chCheckResultPositon": "check_position",  # 僅 EDA 用
    "chResult":           "result",
}

# 日期欄位
BIRTHDAY_COL     = "chBirthday"       # 民國年 YYYMMDD
SCREENING_DATE_COL = "chScreeningDate"  # 民國年 YYYMMDD

# Label 編碼（chResult 為整數型別）
NORMAL_CODE = 0  # chResult == 0 時 result = 0，其餘 = 1

# 欄位編碼設定
GENDER_MAP    = {1: 1, 0: 0}           # chSex 已是 0/1，無需轉換
ABORIGINAL_MAP = {"Y": 1, "N": 0}      # chAboriginal: Y/N → 1/0
# chSmoking、chBinglang 保留為 ordinal 0–5，不轉換
# chOralAwareSymptom 已是 0/1，無需轉換

# 特徵欄位
BASE_FEATURE_COLS = [
    "age", "gender", "screening_year",
    "smoking", "betel_nut",
    "indigenous", "oral_discomfort",
]
DERIVED_FEATURE_COLS = ["screening_count", "prev_result", "years_since_last"]

# 模型訓練
TEST_SIZE    = 0.2
RANDOM_STATE = 42
USE_SMOTE    = True

# 輸出圖表檔名
FIG_LABEL_DIST   = "fig_label_distribution.png"
FIG_YEARLY       = "fig_yearly_trend.png"
FIG_POSITION     = "fig_abnormal_position.png"   # EDA 用
FIG_CONFUSION    = "fig_confusion_matrix.png"
FIG_ROC          = "fig_roc_curve.png"
FIG_IMPORTANCE   = "fig_feature_importance.png"
```

---

## 6. src/preprocess.py — 資料前處理模組

### 6.1 `load_data`
```python
def load_data(path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
```
- **輸入**：CSV 或 Excel 檔案路徑、編碼
- **輸出**：原始 DataFrame
- **行為**：自動判斷副檔名（.csv / .xlsx）並讀取

---

### 6.2 `parse_minguo_date`
```python
def parse_minguo_date(date_val) -> pd.Timestamp | pd.NaT:
```
- **輸入**：民國年日期值（整數或字串，格式 YYYMMDD，如 `0680517` 或 `680517`）
- **輸出**：`pd.Timestamp`（西元日期）或 `pd.NaT`（無效值）
- **行為**：
  - 字串化後補零至 7 位
  - 前 3 碼為民國年，+1911 得西元年
  - 後 4 碼為月日（MMDD）
  - 無法解析時回傳 `pd.NaT`

---

### 6.3 `remove_pii`
```python
def remove_pii(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
```
- **輸入**：原始 DataFrame、識別欄名稱（`chMRNo`）
- **輸出**：去識別化後的 DataFrame，新增整數 `patient_id` 欄
- **行為**：以 `id_col` 分組產生匿名流水號，刪除 `id_col`；同一病患跨年份 ID 一致

---

### 6.4 `compute_age_and_year`
```python
def compute_age_and_year(
    df: pd.DataFrame,
    birthday_col: str,
    screening_date_col: str
) -> pd.DataFrame:
```
- **輸入**：DataFrame、出生日期欄名、篩檢日期欄名（均為民國年格式）
- **輸出**：新增 `age`（int）與 `screening_year`（int）欄的 DataFrame
- **行為**：
  - 呼叫 `parse_minguo_date` 轉換兩個日期欄
  - `age = (screening_date - birthday).days // 365`
  - `screening_year = screening_date.year`
  - 刪除 `birthday_col`（`screening_date_col` 保留至衍生特徵計算後再刪）

---

### 6.5 `rename_columns`
```python
def rename_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
```
- **輸入**：DataFrame、欄位對應字典
- **輸出**：欄位重新命名後的 DataFrame
- **行為**：只重新命名存在的欄位，不存在的靜默略過

---

### 6.6 `encode_columns`
```python
def encode_columns(
    df: pd.DataFrame,
    normal_code: int,
    aboriginal_map: dict
) -> pd.DataFrame:
```
- **輸入**：DataFrame（已重新命名）、正常代碼（整數 `0`）、原住民對應字典
- **輸出**：所有欄位轉為數值的 DataFrame
- **行為**：
  - `result`：`chResult == 0`（整數）→ 0，否則 → 1
  - `indigenous`：依 `aboriginal_map`（Y→1, N→0）轉換；空字串填 0
  - `smoking`、`betel_nut`：已是數值，確認型別後不轉換
  - `gender`、`oral_discomfort`：已是數值，確認型別後不轉換

---

### 6.7 `handle_missing`
```python
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
```
- **輸入**：DataFrame（已編碼）
- **輸出**：缺失值處理後的 DataFrame
- **行為**：
  - 所有欄位先將空字串（`""`）轉為 `NaN`
  - `age`、`screening_year` → 填中位數
  - `smoking`、`betel_nut` → 填 `0`（視為無使用，缺失率 < 1%）
  - `indigenous`、`oral_discomfort` → 填 `0`
  - `gender` → 填眾數
  - `result` 缺失 → 刪除該列（實際資料無缺失）

---

### 6.8 `engineer_features`
```python
def engineer_features(df: pd.DataFrame, screening_date_col: str) -> pd.DataFrame:
```
- **輸入**：DataFrame（含 `patient_id`、篩檢日期欄、`result`）
- **輸出**：新增三個衍生欄位的 DataFrame，並刪除篩檢日期欄
- **新增欄位**：
  - `screening_count`：同一 `patient_id` 截至本筆的累計次數（不含當筆，從 0 起算）
  - `prev_result`：上一筆 `result`（第一次填 -1）
  - `years_since_last`：本次 − 上次篩檢日期（年，第一次填 0）
- **行為**：先依 `patient_id` + 篩檢日期升冪排序再計算

---

### 6.9 `split_and_scale`
```python
def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
```
- **輸入**：特徵矩陣、標籤、測試集比例、隨機種子
- **輸出**：`X_train, X_test, y_train, y_test, scaler`
- **行為**：`stratify=y` 切分，StandardScaler 以 train fit 後同時 transform train/test

---

### 6.10 `apply_smote`
```python
def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
```
- **輸入**：訓練集特徵與標籤
- **輸出**：過採樣後的 `X_train_res, y_train_res`
- **行為**：使用 `imblearn.SMOTE`，只對訓練集操作

---

### 6.11 `run_preprocessing`（主控函式）
```python
def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
```
- **輸入**：無（從 `config.py` 讀取所有設定）
- **輸出**：`X_train, X_test, y_train, y_test, feature_cols`
- **行為**：依序呼叫 6.1 → 6.10，每步驟印出筆數摘要

---

## 7. src/models.py — 模型模組

### 7.1 `build_models`
```python
def build_models(random_state: int) -> dict[str, ClassifierMixin]:
```
- **輸入**：隨機種子
- **輸出**：`{"Random Forest": ..., "Logistic Regression": ..., "SVM": ..., "Decision Tree": ...}`
- **行為**：回傳已設定好參數（但尚未訓練）的模型字典

---

### 7.2 `train_all`
```python
def train_all(
    models: dict[str, ClassifierMixin],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict[str, ClassifierMixin]:
```
- **輸入**：模型字典、訓練資料
- **輸出**：訓練完成的模型字典（同樣結構）
- **行為**：逐一呼叫 `.fit()`，印出每個模型訓練完成的提示

---

## 8. src/evaluate.py — 評估模組

### 8.1 `evaluate_model`
```python
def evaluate_model(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> dict:
```
- **輸入**：單一訓練好的模型、測試資料、模型名稱
- **輸出**：包含以下 key 的字典：
  - `accuracy`, `precision`, `recall`, `f1`, `auc`（float）
  - `y_pred`（ndarray）
  - `y_prob`（ndarray，異常類別機率）
  - `fpr`, `tpr`（ROC curve 用）

---

### 8.2 `evaluate_all`
```python
def evaluate_all(
    trained_models: dict[str, ClassifierMixin],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[dict[str, dict], pd.DataFrame]:
```
- **輸入**：訓練好的模型字典、測試資料
- **輸出**：
  - `results`：每個模型名稱對應的評估字典（由 8.1 產生）
  - `summary_df`：模型比較表（Accuracy/Precision/Recall/F1/AUC）

---

### 8.3 `plot_confusion_matrices`
```python
def plot_confusion_matrices(results: dict, y_test: pd.Series, output_path: str) -> None:
```
- **行為**：2×2 子圖，每個模型一張 Confusion Matrix heatmap，儲存圖片

---

### 8.4 `plot_roc_curves`
```python
def plot_roc_curves(results: dict, output_path: str) -> None:
```
- **行為**：4 條 ROC 曲線畫在同一張圖，含 AUC 標示，儲存圖片

---

### 8.5 `plot_feature_importance`
```python
def plot_feature_importance(
    model: RandomForestClassifier,
    feature_cols: list[str],
    output_path: str
) -> None:
```
- **行為**：橫向長條圖依重要性排序，儲存圖片

---

### 8.6 `plot_abnormal_positions` （EDA 用）
```python
def plot_abnormal_positions(df: pd.DataFrame, output_path: str) -> None:
```
- **輸入**：含 `check_position` 欄的 DataFrame（篩檢異常記錄）
- **輸出**：儲存圖片，無回傳值
- **行為**：解析逗號分隔的位置代號，統計各部位出現頻率，繪製長條圖（簡報加分用）

---

### 8.7 `print_summary`
```python
def print_summary(summary_df: pd.DataFrame) -> None:
```
- **行為**：格式化輸出比較表，標示最佳模型

---

## 9. main.py — 主程式

```python
def main() -> None:
```
執行順序：
1. `preprocess.run_preprocessing()` → 取得訓練/測試資料
2. `models.build_models()` → 建立模型
3. `models.train_all()` → 訓練
4. `evaluate.evaluate_all()` → 評估
5. `evaluate.plot_confusion_matrices()` → 輸出圖
6. `evaluate.plot_roc_curves()` → 輸出圖
7. `evaluate.plot_feature_importance()` → 輸出圖
8. `evaluate.print_summary()` → 印出比較表
9. 儲存 summary CSV 至 `data/processed/`

執行方式：
```bash
source venv/bin/activate
python main.py
```

---

## 10. 資料流總覽

```
data/raw/oral0507-v1.csv
        │
        ▼ load_data()
  原始 DataFrame（含民國年日期、原始代碼）
        │
        ▼ remove_pii()             chMRNo → patient_id，刪除 chMRNo
        │
        ▼ compute_age_and_year()   chBirthday + chScreeningDate → age, screening_year
        │                          刪除 chBirthday
        │
        ▼ rename_columns()         原始欄位名 → 標準英文名
        │
        ▼ encode_columns()         chResult → 0/1，chAboriginal Y/N → 1/0
        │
        ▼ handle_missing()         中位數/眾數/0 填補，刪除 result 缺失列
        │
        ▼ engineer_features()      計算 screening_count, prev_result, years_since_last
        │                          刪除 chScreeningDate
        │
        ▼ split_and_scale()        stratify=y，StandardScaler
  X_train / X_test / y_train / y_test
        │
        ▼ apply_smote()（僅 X_train）
  X_train_res / y_train_res（平衡後）
        │
        ├──▶ train_all() ──▶ evaluate_all()
        │                         │
        │                    ▼ 圖表輸出
        │              fig_confusion_matrix.png
        │              fig_roc_curve.png
        │              fig_feature_importance.png
        │              fig_abnormal_position.png（EDA）
        │              model_comparison.csv
        └──────────────────────────────────────
```

---

## 11. 資料統計摘要（oral0507-v1.csv）

| 項目 | 數值 |
|------|------|
| 資料檔案 | `oral0507-v1.csv` |
| 總筆數 | 21,034 |
| 唯一病患數 | 18,108 |
| 重複記錄（同人同日） | 0（已清理） |
| 正常（chResult=0） | 16,524（78.6%） |
| 異常（chResult≠0） | 4,510（21.4%） |
| 男性比例 | 79.1% |
| 原住民比例 | 35.6% |

### 缺失值摘要

| 欄位 | 缺失率 | 處理方式 |
|------|--------|---------|
| `chSmoking` | 0.7% | 填 0 |
| `chBinglang` | 0.7% | 填 0 |
| `chAboriginal` | 0.5% | 填 "N"（→ 0） |
| `chOralAwareSymptom` | 1.3% | 填 0 |
| `chCheckResultPositon` | 63.8% | 正常（僅異常案例有值） |
| 其他欄位 | 0% | 無需處理 |

---

## 12. 確認事項（✅ 已確認）

| 事項 | 狀態 |
|------|------|
| 資料檔：`oral0507-v1.csv` | ✅ |
| 癌別：口腔癌篩檢 | ✅ |
| Label：`chResult == 0`（整數）為正常，其餘為異常 | ✅ |
| 日期格式：民國年 YYYMMDD（整數） | ✅ |
| 吸菸/檳榔：ordinal 0–5 | ✅ |
| 原住民：Y/N → 1/0（空字串填 0） | ✅ |
| `chCheckResultPositon`：僅 EDA，不入特徵 | ✅ |
| 重複記錄：已清理，無需 deduplicate() | ✅ |
| 空字串需在 handle_missing() 中轉為 NaN | ✅ |
| 資料庫無 `alcohol`、`family_history` 欄位 | ✅ |
