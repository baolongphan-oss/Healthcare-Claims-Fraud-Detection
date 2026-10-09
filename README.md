# Healthcare Claims Fraud Detection

## Project Overview

A machine learning project that predicts which healthcare insurance claims are fraudulent. Fraud makes up only 8.3% of claims, so the project focuses on handling class imbalance, choosing the right evaluation metric, avoiding data leakage, and turning model scores into a cost-based business decision.

**Key Result:** A tuned XGBoost model reaches a **test PR-AUC of 0.798**, about 10x the 0.083 random baseline. At a cost-optimized cutoff, it catches **96% of fraud** while cutting total costs by **87%** compared with paying all fraud and **78%** compared with reviewing every claim.

## Dataset

- **10,000 claims, 20 columns:** patient details, diagnosis and procedure codes, claim and approved amounts, insurance type, provider specialty, filing delay, and a fraud label
- **Fraud rate:** 8.3% (829 claims)
- **Missing values:** 350 each in insurance type, provider specialty, and prior visits

Source: [Healthcare Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset). The notebook downloads it automatically with `kagglehub`.

**Note:** The dataset is synthetic. Several patterns would not appear in real claims data (see Limitations).

## Approach

### 1. Data Cleaning
- **Provider specialty:** All 300 providers have exactly one specialty across their claims, so the 350 missing values were recovered from each provider's other claims rather than guessed.
- **Insurance type:** Missing values filled with "Unknown," since insurance belongs to the patient and the data has no patient ID to link claims.
- **Prior visits:** Filled with the training-set median inside the modeling pipeline, so the test set never influences the fill value.
- Missing values were checked first: fraud rates were nearly identical whether values were missing or present, so the data appears missing at random.

### 2. Leakage Checks
`Claim_Status` and `Approved_Amount` are decided during claim review, after the point where a fraud model would score a claim. Both showed strong leakage (approved claims were 2.2% fraud vs. about 23% for pending and rejected claims), so both were excluded.

### 3. Exploratory Data Analysis
Features were grouped into four categories (patient, visit, claim, and provider) and compared against the 8.3% baseline:
- **Patient and visit features** (age, gender, state, chronic condition, visit type, length of stay, diagnosis, procedure) sat near the baseline.
- **Three billing features stood out:**
  - **Filing delay:** every claim filed within 1 day of service was fraud, and no claim filed after 6 days was
  - **Claim amount:** fraud rose steadily with the billed amount
  - **Provider volume:** providers filing more than about 82 claims a month had much higher fraud rates
- **Diagnosis × procedure pairs:** Fraud appeared most often in the most common pairs, but at near-baseline rates. Unusual rates appeared only in pairs too small to be reliable.
- **Time:** Monthly fraud rates showed no trend or seasonality.

### 4. Modeling
- **Split:** Stratified 70/30 train/test split (7,000 / 3,000 claims), keeping the 8.3% fraud rate in both.
- **Preprocessing:** A scikit-learn `Pipeline` with a `ColumnTransformer` (median imputation and scaling for numeric columns, a log transform for claim amount, and one-hot encoding for categories), fit only on training data.
- **Metric:** PR-AUC, which focuses on the rare fraud class. A random model scores about 0.083.
- **Class imbalance:** Class weighting (`class_weight='balanced'` and XGBoost's `scale_pos_weight`) so fraud mistakes count about 11x as much as normal mistakes.
- **Model comparison:** 5-fold cross-validation on the training set.

| Model | CV PR-AUC |
|---|---|
| **XGBoost** | **0.759** |
| Random Forest | 0.686 |
| Logistic Regression (weighted) | 0.684 |
| Decision Tree | 0.592 |

All four models scored about 0.96 on ROC-AUC, which shows why ROC isn't useful for comparing models on imbalanced data.

- **Tuning:** A grid search found that a slower learning rate (0.03) and fewer trees (200) at depth 3 worked best, raising CV PR-AUC from 0.759 to 0.790.

### 5. Results

**Tuned XGBoost on the test set: PR-AUC 0.798**, close to its cross-validated 0.790, so the model generalizes well to unseen claims.

**Honest check:** Retrained without filing delay, the XGBoost model's PR-AUC dropped from 0.759 to 0.231. Filing delay drives most of the performance, but the remaining features still score almost 3x the baseline, mainly through claim amount and provider volume.

**Permutation importance** (drop in PR-AUC when each feature is shuffled, measured on the untuned XGBoost model):

| Feature | Drop in PR-AUC |
|---|---|
| Filing delay | 0.654 |
| Claim amount | 0.117 |
| Provider volume | 0.032 |
| All other features | about 0 |

Fraud in this dataset is driven by billing behavior, not by who the patient is or what care they received.

### 6. Cost-Based Cutoff
The cutoff was chosen on the training set (using cross-validated probabilities) by minimizing total cost, where each flagged claim costs a review and each missed fraud costs its full claim amount.

| Strategy (test set) | Total Cost |
|---|---|
| Review nothing (pay all fraud) | $263,151 |
| Review every claim | $150,000 |
| **Model at cost-optimized cutoff (0.61)** | **$33,611** |

At the 0.61 cutoff, the model flags 611 of 3,000 claims and catches 96% of fraud at 39% precision. Because a missed fraud claim averages about $1,057, roughly 21 times the review cost, it's worth flagging generously.

## Limitations

- **Synthetic data:** The filing-delay pattern (100% fraud within 1 day, 0% after 6 days) is a hard rule that wouldn't appear in real claims, so real-world performance would be lower. Other signs include an unrealistically uniform length of stay with no 4-day stays, and a retired diagnosis code (M54.5) on claims from 2022 onward.
- **Review cost is an assumption:** The $50 review cost was chosen for illustration, since the data doesn't include investigation costs. In practice, it should come from the fraud team.
- **Features were analyzed one at a time** in EDA, not in combination.

## Future Work

- Test how the best cutoff shifts under different review costs
- Compare against an unsupervised approach (e.g., Isolation Forest) to see how much labeled data matters
- Apply the same workflow to real claims data, where fraud patterns are less clear-cut

## Project Structure

```
.
├── README.md
└── healthcare-fraud-detection.ipynb
```

## Technical Stack

**Tools:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, kagglehub, Jupyter

**Methods:** Missing value analysis, leakage checks, exploratory data analysis, stratified train/test split, scikit-learn Pipelines, class weighting, 5-fold cross-validation, PR-AUC, grid search tuning, permutation importance, cost-based threshold selection

## Contact

- **Author:** Long Phan
- **Email:** lnphan@usc.edu
- **LinkedIn:** [linkedin.com/in/longphan1912](https://www.linkedin.com/in/longphan1912)
- **GitHub:** [github.com/baolongphan-oss](https://github.com/baolongphan-oss)

---

**Last Updated:** October 2026
