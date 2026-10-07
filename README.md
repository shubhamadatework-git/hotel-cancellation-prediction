# Hotel Reservation Cancellation Prediction

Classification model that predicts which hotel bookings will be cancelled, plus recommendations for reducing cancellations. Built as part of a data science On Job Training program, using a public dataset.

**Shubham Adate** | [LinkedIn](https://www.linkedin.com/in/adate-shubham) | [GitHub](https://github.com/shubhamadatework-git)

## Problem
Cancellations hurt revenue and staffing plans. The goal is to predict which bookings are likely to be cancelled and understand what drives it.

## Data
- 36,275 bookings from 2017 to 2018 with booking, guest and stay features (lead time, market segment, room price, special requests and more)
- Public Hotel Reservations dataset (Kaggle). See `data/README.md`.

## Approach
1. Exploratory analysis and cleaning
2. Feature engineering and one-hot encoding
3. Trained a Logistic Regression baseline, a Decision Tree and a Random Forest
4. Compared models on accuracy, precision, recall, F1 and AUC using a held-out test set

## Results (test set)
| Metric | Logistic Regression | Decision Tree | **Random Forest** |
|---|---|---|---|
| Accuracy | 80.4% | 84.8% | **90.2%** |
| Precision | 74.7% | 86.3% | **88.6%** |
| Recall | 63.0% | 65.0% | **81.4%** |
| F1 | 68.4% | 74.2% | **84.8%** |
| AUC | 0.76 | 0.80 | **0.88** |

**Key finding:** lead time (days between booking and arrival) is the biggest driver of cancellations; longer lead times mean higher cancellation risk.

## Recommendations from the analysis
- Dynamic pricing that reflects cancellation risk, especially for long lead times
- Incentives for early confirmation and flexible rebooking options
- Tailored strategies by customer segment, and reminders close to the stay date

## Limitations
- The Random Forest fits the training data much more closely (99% training accuracy) than the test data (90%), so it is somewhat overfit and could be tuned further
- The data covers 2017 to 2018 and may not reflect current booking behaviour
- It may not transfer to other hotels or regions without retraining

## Run it
```bash
pip install -r requirements.txt
jupyter notebook hotel-cancellation-prediction.ipynb
```

**Tech:** Python, pandas, NumPy, scikit-learn, statsmodels, seaborn, matplotlib
