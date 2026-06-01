# 以機器學習預測口腔癌篩檢異常結果

**課程**：115b 機器學習（NDHU）｜**學期**：2025–2026 下學期

利用病患人口特徵、生活習慣風險因子與歷次篩檢紀錄，透過機器學習預測口腔癌篩檢是否出現異常結果（二元分類）。

---

## 資料集

- 來源：花蓮慈濟醫院四癌篩檢資料庫（機構自有資料，去識別化處理）
- 放置路徑：專案根目錄，檔名須與 `config.py` 中 `RAW_DATA_PATH` 一致
- **原始資料不進版控**

---

## 環境建置

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 使用方式

### 1. 訓練所有模型並輸出評估結果

```bash
python main.py
```

執行後會產生：

| 輸出檔案 | 說明 |
|---------|------|
| `data/processed/model_comparison.csv` | 四個模型的 Accuracy / Precision / Recall / F1 / AUC 對照表 |
| `data/processed/fig_confusion_matrix.png` | Confusion Matrix（四模型並排） |
| `data/processed/fig_roc_curve.png` | ROC 曲線比較圖 |
| `data/processed/fig_feature_importance.png` | Random Forest 特徵重要性 |
| `data/processed/fig_abnormal_position.png` | 異常部位分布圖 |
| `data/processed/model_bundle.pkl` | 訓練完成的模型打包檔（供推論使用） |

### 2. 對單一病患進行預測

需先執行過 `main.py` 產生 `model_bundle.pkl`。

**互動式輸入：**

```bash
python predict.py
```

**程式呼叫：**

```python
from predict import predict_patient

result = predict_patient(
    age=45,
    gender=1,           # 1=男, 0=女
    screening_year=2024,
    smoking=3,          # 0=無, 1=已戒, 2–5=現在吸（依強度）
    betel_nut=4,        # 0=無, 1=已戒, 2–5=現在嚼（依強度）
    indigenous=0,       # 1=是, 0=否
    oral_discomfort=1,  # 1=有, 0=無
    screening_count=2,
    prev_result=0,      # 第一次填 -1
    years_since_last=2.0,
)
print(result)
# {'prediction': 1, 'result_label': '異常（建議轉介）', 'abnormal_probability': 0.72, ...}
```

---

## 調整模型參數

所有模型超參數集中在 `config.py`，修改後重新執行 `main.py` 即可：

```python
# config.py
RF_PARAMS  = {"n_estimators": 100, "max_depth": 10, ...}
LR_PARAMS  = {"C": 1.0, "max_iter": 1000, ...}
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", ...}
DT_PARAMS  = {"max_depth": 8, ...}
```

---

## 模型說明

| 模型 | 類型 | 選用原因 |
|------|------|---------|
| Random Forest | 集成學習 | 準確率高；可輸出 Feature Importance |
| Logistic Regression | 線性模型 | 醫療研究常用基準；結果可解釋 |
| SVM | 核函數方法 | 適合二元分類；與其他模型形成對比 |
| Decision Tree | 樹狀模型 | 最易視覺化；樹狀結構直觀呈現決策邏輯 |

---

## 目錄結構

```
final-project/
├── main.py              # 主程式（訓練 + 評估）
├── predict.py           # 推論（單筆預測）
├── config.py            # 路徑、欄位、模型超參數設定
├── requirements.txt
├── data/
│   └── processed/       # 輸出圖表、CSV、model_bundle.pkl
└── src/
    ├── preprocess.py    # 前處理 pipeline
    ├── models.py        # 模型建立與訓練
    └── evaluate.py      # 評估與視覺化
```

---

## 特徵欄位

| 特徵 | 說明 | 類型 |
|------|------|------|
| age | 年齡（由生日與篩檢日期計算） | 數值 |
| gender | 性別（1=男, 0=女） | 二元 |
| screening_year | 篩檢年份 | 數值 |
| smoking | 吸菸強度（0–5） | 有序數值 |
| betel_nut | 嚼檳榔強度（0–5） | 有序數值 |
| indigenous | 是否原住民（1=是, 0=否） | 二元 |
| oral_discomfort | 自覺口腔不適（1=有, 0=無） | 二元 |
| screening_count | 累計篩檢次數（衍生） | 數值 |
| prev_result | 上次篩檢結果（衍生，-1=無紀錄） | 二元 |
| years_since_last | 距上次篩檢年數（衍生） | 數值 |
| **result** | **Label（1=異常, 0=正常）** | 二元 |
# Oral-Cancer-Screening-Abnormality-Prediction
