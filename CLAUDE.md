# 機器學習期末專題 - 開發目錄

## 課程資訊
- 課程：115b 機器學習（NDHU）
- 學期：2025–2026 下學期

## 專案說明
- 題目：以機器學習預測大腸癌篩檢異常結果
- 資料集來源：醫院四癌篩檢歷年資料庫（自行提供，去識別化後使用）
- Label：篩檢結果 1=異常 / 0=正常（二元分類）
- 使用模型（至少 3 個）：Random Forest、Logistic Regression、SVM（+ 可選 Decision Tree）

## 環境
- Python，虛擬環境位於 `venv/`
- 啟動：`source venv/bin/activate`
- 安裝套件：`pip install -r requirements.txt`

## 目錄結構
```
final-project/
├── CLAUDE.md
├── requirements.txt
├── venv/                   # 不進版控
├── data/
│   ├── raw/                # 原始資料（不進版控）
│   └── processed/          # 處理後資料
├── notebooks/
│   ├── 01_eda.ipynb        # 探索性資料分析
│   ├── 02_preprocessing.ipynb  # 前處理 pipeline
│   └── 03_modeling.ipynb   # 模型訓練與比較
└── src/
    ├── preprocess.py       # 前處理函式
    └── models.py           # 模型訓練函式
```

## 欄位設計（原始欄位 → 標準名稱）

| 標準名稱 | 原始欄位 | 說明 | 類型 |
|---------|---------|------|------|
| patient_id | chMRNo | 去識別化流水號 | int |
| age | chBirthday + chScreeningDate | 年齡（計算） | int |
| screening_year | chScreeningDate | 篩檢年份（計算） | int |
| gender | chSex | 性別 1=男, 0=女 | int |
| smoking | chSmoking | 吸菸強度 0–5 | int |
| betel_nut | chBinglang | 嚼檳榔強度 0–5 | int |
| indigenous | chAboriginal | 是否原住民 1=是, 0=否 | int |
| oral_discomfort | chOralAwareSymptom | 自覺口腔不適 1=有, 0=無 | int |
| screening_count | 衍生 | 累計篩檢次數 | int |
| prev_result | 衍生 | 上次篩檢結果（-1=無紀錄） | int |
| years_since_last | 衍生 | 距上次篩檢年數 | float |
| result | chResult | **Label** 1=異常, 0=正常 | int |

## 給 Claude 的注意事項
- 工作範圍限於此目錄，不要修改上層 Obsidian 筆記
- 上層目錄 `115b-machine-learning/` 存放課程筆記（.md），勿更動
- 整個 `/Users/peter/Obsidian/PeterNotes` 是 Obsidian vault，其他子目錄是別的課程
