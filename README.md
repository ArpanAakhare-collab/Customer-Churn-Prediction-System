# Customer Churn Prediction

A supervised machine-learning mini-project that predicts whether a telecom customer may churn. The notebook performs EDA, preprocessing, feature engineering, model comparison, evaluation, and saves a fitted scikit-learn pipeline. The Streamlit app uses that saved pipeline to predict churn for a user-entered customer profile.

## Project files

- `Customer_Churn_Prediction_Colab.ipynb` — the Google Colab training and evaluation notebook (add this file; it may be placed in a `notebooks/` folder).
- `app.py` — Streamlit user interface.
- `customer_churn_model.joblib` — trained pipeline downloaded from the notebook. Keep this beside `app.py`.
- `requirements.txt` — dependencies for the deployed app.

## Dataset

IBM Telco Customer Churn sample dataset: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Before deploying, pin scikit-learn and compatible dependencies in `requirements.txt` to match the versions used to train the saved model in Colab. See `DEPLOYMENT_GUIDE.md` for the full deployment steps.

## Important limitation

Predictions are estimates based on a historical sample dataset, not guarantees about real customers. Do not use them as the sole basis for real business decisions.
