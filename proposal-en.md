# Machine Learning Final Project Proposal

---

## 1. Group Member Names

- Name: Chien-Wei Tseng
- Student ID: 611421501
- Email: 611421501@gms.ndhu.edu.tw

---

## 2. Project Title

**Oral Cancer Screening Abnormality Prediction Using Machine Learning**

---

## 3. Project Motivation

Oral cancer has long ranked among the top five cancers in Taiwanese males. Unlike many other cancers, oral cancer is strongly associated with preventable lifestyle habits — particularly betel nut chewing and smoking, both of which are prevalent in Taiwan.

Research indicates that smokers face a 3× higher risk of oral cancer compared to non-smokers, while betel nut chewers face a 5× higher risk. Those who both smoke and chew betel nut may face a risk tens of times higher than the general population. Although the government provides free oral cancer screening programs targeting high-risk individuals, screening participation rates remain insufficient, and many high-risk individuals do not receive timely examinations.

This project aims to leverage machine learning to **predict which individuals are likely to receive abnormal results in oral cancer screenings**, based on demographic characteristics, lifestyle risk factors, and historical screening records. The practical value lies in enabling healthcare providers to proactively contact high-risk individuals, encourage participation in screenings, and ultimately improve early detection rates and patient outcomes.

---

## 4. Proposed Methods

This project plans to compare the following four machine learning models (at least 3 recommended):

| Model | Reason for Selection |
|-------|---------------------|
| **Random Forest** | Ensemble learning with typically high accuracy; outputs Feature Importance to reveal which risk factors contribute most to predictions |
| **Logistic Regression** | Commonly used baseline model in medical research; highly interpretable results that clearly show each factor's contribution to risk |
| **SVM (Support Vector Machine)** | Kernel-based method well-suited for binary classification; provides contrast with linear and tree-based models |
| **Decision Tree** | Most intuitive and interpretable model; tree structure can be visually presented in slides to illustrate decision logic |

The four models belong to different algorithm categories — ensemble learning, linear model, kernel method, and tree-based model — enabling meaningful cross-category comparison.

---

## 5. Dataset Source

- **Source**: Hualien Tzu Chi Hospital Four-Cancer Screening Database (institutional data, not publicly available)
- **Nature**: Self-collected screening records
- **Time Range**: Oral cancer screening data spanning the past 10 years
- **Number of Instances**: 10,000+
- **Number of Attributes**: 10 features + 1 label

### Feature Description

| Feature | Description | Type |
|---------|-------------|------|
| Age | Calculated from date of birth and screening date | Numerical |
| Gender | Male / Female | Binary |
| Screening Year | Year of the current screening | Numerical |
| Smoking Intensity | 0=Never; 1=Quit; 2–5=Current smoker (graded by years and daily amount) | Ordinal |
| Betel Nut Intensity | 0=Never; 1=Quit; 2–5=Current chewer (graded by years and daily amount) | Ordinal |
| Indigenous | Whether the patient is indigenous (Yes / No) | Binary |
| Oral Discomfort | Self-reported oral discomfort (Yes / No) | Binary |
| Screening Count | Cumulative number of screenings up to the current visit (derived) | Numerical |
| Previous Result | Result of the most recent prior screening (derived; -1 if no prior record) | Binary |
| Years Since Last | Years elapsed since the previous screening (derived; 0 if first visit) | Numerical |

### Label

| Value | Meaning | Original Code |
|-------|---------|---------------|
| 0 | Normal | `00` (no significant abnormality) |
| 1 | Abnormal | Other codes (e.g., leukoplakia, erythroplakia, suspected oral cancer) |

> Note: All data will be de-identified prior to use, replacing personal identifiers with anonymized serial numbers.

---

## 6. Expected Results

1. **Model Performance Comparison**: Evaluate all four models using Accuracy, Precision, Recall, F1-score, and AUC to identify the most suitable algorithm for predicting oral cancer screening abnormalities.

2. **Feature Importance Analysis**: Use Random Forest Feature Importance to identify the most influential factors (e.g., betel nut intensity, smoking intensity, previous screening result).

3. **Risk Factor Insights**: Analyze the gradient relationship between smoking/betel nut intensity levels (0–5) and abnormal screening rates, expecting to observe a pattern of "higher intensity → higher positive rate."

4. **Visualization Outputs**: Confusion Matrix for each model, ROC curve comparison chart, Feature Importance bar chart, and abnormal lesion location distribution (e.g., frequency of findings on the tongue, buccal mucosa, etc.).

5. **Practical Contribution**: Demonstrate that basic demographic and lifestyle data alone can effectively predict oral cancer screening outcomes, supporting healthcare providers in building a proactive high-risk case referral system.
