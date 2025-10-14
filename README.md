# Spotify_Churn_Prediction (2025 Dataset)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![Scikit-learn](https://img.shields.io/badge/Scikit-learn-1.2%2B-yellow)](https://scikit-learn.org/) [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/YOUR_USERNAME/spotify-churn-prediction) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

Predicting whether Spotify users will churn (cancel their subscription) using machine learning on a 2025 dataset. This project explores exploratory data analysis (EDA), handles class imbalance, and benchmarks models like Logistic Regression, Decision Trees, Random Forest, and XGBoost.

Inspired by real-world churn analytics—think: "Why did that Premium user dip after bingeing playlists?"

## 📊 Project Overview
- **Goal**: Build a binary classifier to predict `is_churned` (0: Active, 1: Churned) based on user behavior and demographics.
- **Key Insights**:
  - Churn rate: ~20-25% (imbalanced dataset—handled via undersampling).
  - Top predictors: `subscription_type`, `ads_listened_per_week`, `listening_time`.
  - Best model: Decision Tree (Recall ~50% for churners—great for early detection despite false positives).
- **Challenges**: Modest performance (ROC-AUC ~0.50-0.53) due to noisy data; suggests need for more features (e.g., playlist diversity) in production.

Full results table (from models):
| Model              | Accuracy | Precision | Recall  | F1-Score | ROC-AUC |
|--------------------|----------|-----------|---------|----------|---------|
| Logistic Regression| 0.518   | 0.266    | 0.490  | 0.345   | 0.497  |
| Decision Tree     | 0.515   | 0.267    | 0.502  | 0.349   | 0.501  |
| Random Forest     | 0.725   | 0.297    | 0.046  | 0.079   | 0.530  |
| XGBoost           | 0.608   | 0.291    | 0.357  | 0.321   | 0.518  |

## 🗄️ Dataset
- **Source**: Synthetic 2025 Spotify user data (8,000+ rows).
- **Columns**:
  - `user_id`: Unique ID.
  - `gender`: Male/Female/Other.
  - `age`: User age (16-59).
  - `country`: e.g., CA, US, DE.
  - `subscription_type`: Free/Family/Premium/Student.
  - `listening_time`: Minutes per session.
  - `songs_played_per_day`: Daily plays.
  - `skip_rate`: Fraction of songs skipped (0-1).
  - `device_type`: Desktop/Mobile/Web.
  - `ads_listened_per_week`: Ad exposures.
  - `offline_listening`: 0/1 (offline mode usage).
  - `is_churned`: Target (0/1).
- Download: `spotify_churn_dataset.csv` (included in repo).

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **Libraries**: Pandas, NumPy, Matplotlib/Seaborn (EDA), Scikit-learn (ML), SciPy (stats).
- **Environment**: Jupyter Notebook.

## 🚀 Quick Start
1. **Clone the Repo**:
