RAW_DATA_PATH = "oral0507-v1.csv"
PROCESSED_DIR = "data/processed/"

PATIENT_ID_COL = "chMRNo"
BIRTHDAY_COL = "chBirthday"
SCREENING_DATE_COL = "chScreeningDate"

RENAME_MAP = {
    "chSex":               "gender",
    "chSmoking":           "smoking",
    "chBinglang":          "betel_nut",
    "chAboriginal":        "indigenous",
    "chOralAwareSymptom":  "oral_discomfort",
    "chCheckResultPositon":"check_position",
    "chResult":            "result",
}

NORMAL_CODE = 0
ABORIGINAL_MAP = {"Y": 1, "N": 0}

BASE_FEATURE_COLS = [
    "age", "gender",
    "smoking", "betel_nut",
    "indigenous", "oral_discomfort",
]
DERIVED_FEATURE_COLS = ["screening_count", "prev_result", "years_since_last"]

TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_SMOTE = True

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "class_weight": "balanced",
    "n_jobs": -1,
}
LR_PARAMS = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
}
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "rbf",
    "class_weight": "balanced",
    "probability": True,
}
DT_PARAMS = {
    "max_depth": 8,
    "class_weight": "balanced",
}

FIG_LABEL_DIST  = "fig_label_distribution.png"
FIG_YEARLY      = "fig_yearly_trend.png"
FIG_POSITION    = "fig_abnormal_position.png"
FIG_CONFUSION   = "fig_confusion_matrix.png"
FIG_ROC         = "fig_roc_curve.png"
FIG_IMPORTANCE  = "fig_feature_importance.png"
