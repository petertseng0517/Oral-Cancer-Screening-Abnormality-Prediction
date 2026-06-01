import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import config


def load_data(path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding=encoding, dtype=str)
    print(f"[load_data] 載入 {len(df):,} 筆，{df.shape[1]} 欄")
    return df


def parse_minguo_date(date_val) -> pd.Timestamp:
    try:
        s = str(date_val).strip().zfill(7)
        year = int(s[:3]) + 1911
        month = int(s[3:5])
        day = int(s[5:7])
        return pd.Timestamp(year, month, day)
    except Exception:
        return pd.NaT


def remove_pii(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df["patient_id"] = pd.factorize(df[id_col])[0]
    df = df.drop(columns=[id_col])
    print(f"[remove_pii] 唯一病患數：{df['patient_id'].nunique():,}")
    return df


def compute_age_and_year(
    df: pd.DataFrame,
    birthday_col: str,
    screening_date_col: str,
) -> pd.DataFrame:
    df = df.copy()
    birthday = df[birthday_col].apply(parse_minguo_date)
    screening = df[screening_date_col].apply(parse_minguo_date)
    df["age"] = ((screening - birthday).dt.days // 365).astype("Int64")
    df["screening_year"] = screening.dt.year.astype("Int64")
    df = df.drop(columns=[birthday_col])
    print(f"[compute_age_and_year] age 範圍：{df['age'].min()}–{df['age'].max()}")
    return df


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def encode_columns(
    df: pd.DataFrame,
    normal_code: int,
    aboriginal_map: dict,
) -> pd.DataFrame:
    df = df.copy()
    df["result"] = df["result"].apply(
        lambda x: 0 if str(x).strip() == str(normal_code) else 1
    ).astype(int)
    df["indigenous"] = (
        df["indigenous"].astype(str).str.strip().map(aboriginal_map).fillna(0).astype(int)
    )
    for col in ["smoking", "betel_nut", "oral_discomfort", "gender"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["result"])
    for col in ["age", "screening_year"]:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)
    for col in ["smoking", "betel_nut", "oral_discomfort", "indigenous"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna(df["gender"].mode()[0])
    before = len(df)
    df = df.dropna(subset=config.BASE_FEATURE_COLS)
    print(f"[handle_missing] 處理後：{len(df):,} 筆（刪除 {before - len(df)} 筆）")
    return df


def engineer_features(df: pd.DataFrame, screening_date_col: str) -> pd.DataFrame:
    df = df.copy()
    screening_dt = df[screening_date_col].apply(parse_minguo_date)
    df["_screening_dt"] = screening_dt
    df = df.sort_values(["patient_id", "_screening_dt"]).reset_index(drop=True)

    df["screening_count"] = df.groupby("patient_id").cumcount()

    df["prev_result"] = (
        df.groupby("patient_id")["result"]
        .shift(1)
        .fillna(-1)
        .astype(int)
    )

    prev_dt = df.groupby("patient_id")["_screening_dt"].shift(1)
    df["years_since_last"] = (
        (df["_screening_dt"] - prev_dt).dt.days / 365.25
    ).fillna(0).round(2)

    df = df.drop(columns=[screening_date_col, "_screening_dt"])
    print(f"[engineer_features] 衍生特徵計算完成，共 {len(df):,} 筆")
    return df


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
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


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple:
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[apply_smote] SMOTE 後 — 正常：{(y_res==0).sum():,}  異常：{(y_res==1).sum():,}")
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)


def run_preprocessing() -> tuple:
    df = load_data(config.RAW_DATA_PATH)
    df = remove_pii(df, config.PATIENT_ID_COL)
    df = compute_age_and_year(df, config.BIRTHDAY_COL, config.SCREENING_DATE_COL)
    df = rename_columns(df, config.RENAME_MAP)
    df = encode_columns(df, config.NORMAL_CODE, config.ABORIGINAL_MAP)
    df = handle_missing(df)
    df = engineer_features(df, config.SCREENING_DATE_COL)

    feature_cols = config.BASE_FEATURE_COLS + config.DERIVED_FEATURE_COLS
    X = df[feature_cols].astype(float)
    y = df["result"].astype(int)

    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        X, y, config.TEST_SIZE, config.RANDOM_STATE
    )

    if config.USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train, config.RANDOM_STATE)

    return X_train, X_test, y_train, y_test, feature_cols, scaler
