![Spotify Logo](https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg)

# Spotify User Churn Prediction (2025 Dataset)

## 📖 Overview
This project analyzes Spotify user behavior to predict **churn** (whether a user cancels their subscription or remains active). Using a dataset of ~8,000 users, we perform Exploratory Data Analysis (EDA), preprocess features, and train/evaluate multiple machine learning models. The goal? Help Spotify retain users by identifying at-risk ones early.

Key insights:
- **Churn Rate**: ~20-25% (imbalanced classes—churners are the minority).
- **Top Predictors**: Subscription type, listening time, and ads exposure.
- **Best Model**: Logistic Regression for high recall (~52% on churners); therefore Logistic Regression for balanced performance.

## 📊 Dataset
- **Source**: Synthetic Spotify churn data (2025 edition) with 8,000+ rows.
- **Features**:
  | Column                  | Description                          | Type    | Example Values          |
  |-------------------------|--------------------------------------|---------|-------------------------|
  | `user_id`              | Unique user identifier              | int    | 1, 2, ...              |
  | `gender`               | User gender                         | str    | Female, Male, Other    |
  | `age`                  | User age                            | int    | 17-59                  |
  | `country`              | User country code                   | str    | CA, US, DE, AU, ...    |
  | `subscription_type`    | Plan type                           | str    | Free, Premium, Family, Student |
  | `listening_time`       | Weekly listening hours              | int    | 0-300                  |
  | `songs_played_per_day` | Daily songs played                  | int    | 0-100                  |
  | `skip_rate`            | Fraction of songs skipped           | float  | 0.0-1.0                |
  | `device_type`          | Primary device                      | str    | Mobile, Desktop, Web   |
  | `ads_listened_per_week`| Weekly ad exposures (Free users)    | int    | 0-50                   |
  | `offline_listening`    | Offline mode usage (0/1)            | int    | 0, 1                   |
  | `is_churned`           | Target: 1 if churned, 0 if active   | int    | 0, 1                   |

- **Download**: [spotify_churn_dataset.csv](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/data) (included in repo).
- **Size**: ~8,000 rows × 12 columns.
- **Challenges**: Class imbalance (80% active, 20% churned) → Handled with undersampling in `UnderSampling.ipynb`.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Libraries**:
  - Data: `pandas`, `numpy`, `seaborn`, `matplotlib`
  - ML: `scikit-learn` (LogisticRegression, DecisionTree, RandomForest), `xgboost`
  - Utils: `scipy` for stats
- **Environment**: Jupyter Notebook (tested on Kaggle/Colab).


(Key ones: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, scipy)

3. Run notebooks:
- Open in Jupyter: `jupyter notebook`
- Or Colab: Upload and run.

## 🔍 Analysis & Models
### 1. EDA (Exploratory Data Analysis)
- Visualizations: Distributions, correlations, churn by demographics/subscription.
- Insights: Free users churn more (high ads/skip rate); Premium users with low listening time at risk.
- Notebook: [spotify-2025-eda-prediction-models.ipynb](spotify-2025-eda-prediction-models.ipynb)

### 2. Preprocessing
- Encoding: One-hot for categoricals (gender, country, etc.).
- Scaling: StandardScaler for numerics.
- Train/Test Split: 80/20 stratified.

### 3. Models & Performance
Trained on balanced data; evaluated with Accuracy, Precision, Recall, F1, ROC-AUC.

| Model              | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|--------------------|----------|-------------------|----------------|------------|---------|
| Logistic Regression| 0.518   | 0.266            | 0.490         | 0.345     | 0.497  |
| Decision Tree     | 0.515   | 0.267            | 0.502         | 0.349     | 0.501  |
| Random Forest     | 0.725   | 0.297            | 0.046         | 0.079     | 0.530  |
| XGBoost           | 0.608   | 0.291            | 0.357         | 0.321     | 0.518  |

- **Threshold Tuning**: In `spotify_copy1.ipynb`, we tuned to 0.46 for better recall (~43% on churners).
- **Visuals**: Confusion matrices, feature importances, ROC curves.

### Key Findings
- **Top Features**: `subscription_type`, `listening_time`, `ads_listened_per_week`.
- **Recommendations**: Target Free users with high skips/ads via personalized offers.
- **Limitations**: Modest recall (32-50%)—future: Ensemble more models or add time-series features (e.g., session duration trends).

## 🎯 Conclusions
- **For Retention**: Use Decision Tree to flag ~50% of churners (tolerate false positives for outreach).
- **Balanced Use**: Logistic for stable predictions.
- Models aren't production-ready yet—improve with more data or advanced techniques like SHAP for explainability.

## 🚀 Next Steps & Improvements
- Will try oversampling and undersampling to see whether the performance boosts or not.
- Add cross-validation and hyperparameter tuning (e.g., GridSearchCV on Random Forest).
- Deploy: Streamlit app for predictions.
- Extend: Incorporate audio features (e.g., valence from Spotify API).


## 📄 License
MIT License—use freely, just credit back.

## 👏 Acknowledgments
- Dataset inspired by Kaggle Spotify challenges.
- Built with ❤️ by [Shivam Rawat](https://github.com/shivamrawat03).
