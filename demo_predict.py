"""
口腔癌篩檢異常結果預測 — 單筆推論 Demo
用於簡報現場展示 / 錄影：輸入病患特徵，輸出風險判定與異常機率
準備兩組對照案例（低風險 → 正常／高風險 → 異常），呈現模型的判別能力
"""
from predict import predict_patient


def show_case(title, description, **features):
    print("=" * 50)
    print(f"【{title}】")
    print(f"  {description}")
    print("-" * 50)

    result = predict_patient(**features)

    print(f"  預測結果：{result['result_label']}")
    print(f"  異常機率：{result['abnormal_probability'] * 100:.1f}%")
    print(f"  判定門檻：{result['threshold']}（由預設 0.5 調降，提高 Recall）")
    print(f"  使用模型：{result['model_used']}")
    print("=" * 50)
    print()


show_case(
    "案例一：低風險／正常",
    "55 歲、女性，不吸菸、嚼檳榔已戒、無自覺口腔不適，\n  過去已篩檢 3 次且結果皆正常、距上次篩檢已 6 年",
    age=55, gender=0, smoking=0, betel_nut=1, indigenous=0,
    oral_discomfort=0, screening_count=3, prev_result=0, years_since_last=6.0,
)

show_case(
    "案例二：高風險／異常",
    "45 歲、男性，現有吸菸與嚼檳榔習慣，\n  自覺口腔不適，且上次篩檢曾有異常紀錄",
    age=45, gender=1, smoking=4, betel_nut=5, indigenous=0,
    oral_discomfort=1, screening_count=0, prev_result=1, years_since_last=0.0,
)
