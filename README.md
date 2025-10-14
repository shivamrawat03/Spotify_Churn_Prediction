![Spotify Logo](https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg)

# Spotify User Churn Prediction (2025 Dataset)

## 📖 Overview
This project analyzes Spotify user behavior to predict **churn** (whether a user cancels their subscription or remains active). Using a dataset of ~8,000 users, we perform Exploratory Data Analysis (EDA), preprocess features, and train/evaluate multiple machine learning models. The goal? Help Spotify retain users by identifying at-risk ones early.

Key insights:
- **Churn Rate**: ~20-25% (imbalanced classes—churners are the minority).
- **Top Predictors**: Subscription type, listening time, and ads exposure.
- **Best Model**: Decision Tree for high recall (~50% on churners) if false positives are okay; XGBoost for balanced performance.

Full analysis and dashboard available on [GitHub](https://github.com/YourUsername/Spotify-Churn-Prediction). (Update this to your repo link!)

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

- **Download**: [spotify_churn_dataset.csv](spotify_churn_dataset.csv) (included in repo).
- **Size**: ~8,000 rows × 12 columns.
- **Challenges**: Class imbalance (80% active, 20% churned) → Handled with undersampling in `UnderSampling.ipynb`.

## 🛠️ Tech Stack
- **Language**: Python 3.9+
- **Libraries**:
  - Data: `pandas`, `numpy`, `seaborn`, `matplotlib`
  - ML: `scikit-learn` (LogisticRegression, DecisionTree, RandomForest), `xgboost`
  - Utils: `scipy` for stats
- **Environment**: Jupyter Notebook (tested on Kaggle/Colab).

### Installation
1. Clone the repo:
