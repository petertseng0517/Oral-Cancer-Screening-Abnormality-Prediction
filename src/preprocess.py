import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import config


# ──────────────────────────────────────────────────────────
# 1. 載入資料
# ──────────────────────────────────────────────────────────

def load_data(path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    # 支援 Excel（.xlsx）或 CSV 格式，全部欄位以字串讀入避免型別自動轉換錯誤
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding=encoding, dtype=str)
    print(f"[load_data] 載入 {len(df):,} 筆，{df.shape[1]} 欄")
    return df


# ──────────────────────────────────────────────────────────
# 2. 日期工具函式：民國年 → 西元年
# ──────────────────────────────────────────────────────────

def parse_minguo_date(date_val) -> pd.Timestamp:
    # 醫院系統的日期格式為民國年（7位數字，如 1020305 = 民國102年3月5日）
    # 前3碼為民國年，加1911換算成西元年
    try:
        s = str(date_val).strip().zfill(7)
        year = int(s[:3]) + 1911
        month = int(s[3:5])
        day = int(s[5:7])
        return pd.Timestamp(year, month, day)
    except Exception:
        return pd.NaT  # 無法解析時回傳空值


# ──────────────────────────────────────────────────────────
# 3. 去識別化
# ──────────────────────────────────────────────────────────

def remove_pii(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    # 將原始病歷號（個人識別資料）以匿名流水號取代
    # pd.factorize() 會將每個唯一值對應到一個整數，原始 ID 欄位隨即刪除
    df = df.copy()
    df["patient_id"] = pd.factorize(df[id_col])[0]
    df = df.drop(columns=[id_col])
    print(f"[remove_pii] 唯一病患數：{df['patient_id'].nunique():,}")
    return df


# ──────────────────────────────────────────────────────────
# 4. 計算年齡與篩檢年份（衍生數值特徵）
# ──────────────────────────────────────────────────────────

def compute_age_and_year(
    df: pd.DataFrame,
    birthday_col: str,
    screening_date_col: str,
) -> pd.DataFrame:
    df = df.copy()
    # 將民國年格式的生日與篩檢日期都轉換成 pd.Timestamp
    birthday = df[birthday_col].apply(parse_minguo_date)
    screening = df[screening_date_col].apply(parse_minguo_date)
    # 年齡 = (篩檢日期 - 生日) ÷ 365，取整數天數
    df["age"] = ((screening - birthday).dt.days // 365).astype("Int64")
    # 篩檢年份直接從篩檢日期取出西元年
    df["screening_year"] = screening.dt.year.astype("Int64")
    df = df.drop(columns=[birthday_col])
    print(f"[compute_age_and_year] age 範圍：{df['age'].min()}–{df['age'].max()}")
    return df


# ──────────────────────────────────────────────────────────
# 5. 欄位重新命名（原始欄位 → 標準名稱）
# ──────────────────────────────────────────────────────────

def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    # 只重命名資料中實際存在的欄位，避免 KeyError
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


# ──────────────────────────────────────────────────────────
# 6. 類別編碼（將文字類別轉為數值）
# ──────────────────────────────────────────────────────────

def encode_columns(
    df: pd.DataFrame,
    normal_code: int,
    aboriginal_map: dict,
) -> pd.DataFrame:
    df = df.copy()
    # Label 編碼：篩檢結果代碼 '00' = 正常(0)，其他代碼 = 異常(1)
    df["result"] = df["result"].apply(
        lambda x: 0 if str(x).strip() == str(normal_code) else 1
    ).astype(int)
    # 原住民欄位：'Y' → 1，'N' → 0
    df["indigenous"] = (
        df["indigenous"].astype(str).str.strip().map(aboriginal_map).fillna(0).astype(int)
    )
    # 其餘數值型欄位（吸菸、檳榔、性別、口腔不適）轉為數值，無法轉換的填 NaN
    for col in ["smoking", "betel_nut", "oral_discomfort", "gender"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ──────────────────────────────────────────────────────────
# 7. 缺漏值處理
# ──────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Label 缺漏無法填補，直接刪除該筆資料
    df = df.dropna(subset=["result"])
    # 數值型特徵（年齡、篩檢年份）用中位數填補，避免極端值影響
    for col in ["age", "screening_year"]:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    # 生活習慣類特徵缺漏視為「無此習慣」，填 0
    for col in ["smoking", "betel_nut", "oral_discomfort", "indigenous"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # 性別用眾數（最常出現的值）填補
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
    before = len(df)
    # 刪除基本特徵仍有缺漏的資料列
    df = df.dropna(subset=config.BASE_FEATURE_COLS)
    print(f"[handle_missing] 處理後：{len(df):,} 筆（刪除 {before - len(df)} 筆）")
    return df


# ──────────────────────────────────────────────────────────
# 8. 衍生特徵工程（病患歷史資訊）
# ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, screening_date_col: str) -> pd.DataFrame:
    df = df.copy()
    screening_dt = df[screening_date_col].apply(parse_minguo_date)
    df["_screening_dt"] = screening_dt
    # 依照病患 ID 與篩檢日期排序，確保歷史紀錄順序正確
    df = df.sort_values(["patient_id", "_screening_dt"]).reset_index(drop=True)

    # 累計篩檢次數：每位病患第 1 次為 0，第 2 次為 1，依此類推（cumcount 從 0 開始）
    df["screening_count"] = df.groupby("patient_id").cumcount()

    # 上次篩檢結果：取同一病患的前一筆 result，初次篩檢（無前紀錄）填 -1
    df["prev_result"] = (
        df.groupby("patient_id")["result"]
        .shift(1)
        .fillna(-1)
        .astype(int)
    )

    # 距上次篩檢年數：計算本次與前一次篩檢日期的間隔天數，再除以 365.25 換算為年
    prev_dt = df.groupby("patient_id")["_screening_dt"].shift(1)
    df["years_since_last"] = (
        (df["_screening_dt"] - prev_dt).dt.days / 365.25
    ).fillna(0).round(2)  # 初次篩檢無前紀錄，填 0

    # 移除計算用的輔助欄位
    df = df.drop(columns=[screening_date_col, "_screening_dt"])
    print(f"[engineer_features] 衍生特徵計算完成，共 {len(df):,} 筆")
    return df


# ──────────────────────────────────────────────────────────
# 9. 資料切割 + 標準化
# ──────────────────────────────────────────────────────────

def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple:
    # 訓練/測試集切割（預設 80/20）
    # stratify=y：確保兩組的陽性（異常）比例與原始資料一致，避免隨機偏差
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # 標準化（Z-score normalization）：讓每個特徵的均值為 0、標準差為 1
    # 注意：scaler 只在訓練集上 fit，再套用到測試集，避免資料洩漏（data leakage）
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    print(f"[split_and_scale] Train：{len(X_train):,}  Test：{len(X_test):,}")
    print(f"  Train 異常率：{y_train.mean()*100:.1f}%  Test 異常率：{y_test.mean()*100:.1f}%")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ──────────────────────────────────────────────────────────
# 10. SMOTE 過採樣（處理類別不平衡）
# ──────────────────────────────────────────────────────────

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple:
    # 口腔癌篩檢陽性率偏低，訓練集中「正常」遠多於「異常」
    # SMOTE（Synthetic Minority Over-sampling Technique）：
    #   對少數類別（異常）合成新的假資料，使兩類數量平衡
    #   只對訓練集做，測試集保持原始分布才能正確評估真實效能
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[apply_smote] SMOTE 後 — 正常：{(y_res==0).sum():,}  異常：{(y_res==1).sum():,}")
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


# ──────────────────────────────────────────────────────────
# 主流程：依序執行以上所有步驟
# ──────────────────────────────────────────────────────────

def run_preprocessing() -> tuple:
    df = load_data(config.RAW_DATA_PATH)               # 1. 載入
    df = remove_pii(df, config.PATIENT_ID_COL)         # 3. 去識別化
    df = compute_age_and_year(                         # 4. 計算年齡與年份
        df, config.BIRTHDAY_COL, config.SCREENING_DATE_COL
    )
    df = rename_columns(df, config.RENAME_MAP)         # 5. 重命名
    df = encode_columns(df, config.NORMAL_CODE, config.ABORIGINAL_MAP)  # 6. 編碼
    df = handle_missing(df)                            # 7. 缺漏值處理
    df = engineer_features(df, config.SCREENING_DATE_COL)  # 8. 衍生特徵

    # 組合最終特徵矩陣 X 與標籤向量 y
    feature_cols = config.BASE_FEATURE_COLS + config.DERIVED_FEATURE_COLS
    X = df[feature_cols].astype(float)
    y = df["result"].astype(int)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(  # 9. 切割 + 標準化
        X, y, config.TEST_SIZE, config.RANDOM_STATE
    )

    if config.USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train, config.RANDOM_STATE)  # 10. SMOTE

    return X_train, X_test, y_train, y_test, feature_cols, scaler
