# Customer Churn Prediction — Web App Deployment

## 1. Train the model in Google Colab

1. Open `Customer_Churn_Prediction_Colab.ipynb` in Google Colab.
2. Select **Runtime → Run all** and wait for the notebook to finish.
3. In the final model-save cell, download `customer_churn_model.joblib` to your computer.
4. Run this additional Colab cell and note the versions printed; use the exact scikit-learn (and preferably numpy/joblib) versions in `requirements.txt` to avoid model-serialization compatibility issues:

```python
import platform, sklearn, numpy, pandas, joblib
print("Python:", platform.python_version())
print("scikit-learn:", sklearn.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("joblib:", joblib.__version__)
```

For example, replace dependency lines with the exact versions you see, e.g. `scikit-learn==YOUR_VERSION`. Do not literally leave `YOUR_VERSION` in the file.

## 2. Prepare the GitHub repository

Create a public GitHub repository named `customer-churn-prediction` and upload these files at the repository root:

- `app.py` (from this folder)
- `requirements.txt` (after checking/pinning versions)
- `customer_churn_model.joblib` (downloaded from Colab)
- `Customer_Churn_Prediction_Colab.ipynb` (your notebook; you can put it in a `notebooks/` folder)
- `README.md` (describe the project, dataset source, algorithms, metrics, and how to run it)

The model file must be beside `app.py` because the app looks for it in the same directory. The CSV is not required by the deployed app; it is used in the notebook to train/evaluate the model.

## 3. Deploy using Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose the `customer-churn-prediction` repository, the `main` branch, and `app.py` as the entrypoint.
4. In **Advanced settings**, select a Python version compatible with the version you noted from Colab, when available.
5. Click **Deploy** and wait for the build to finish.
6. Open the generated `*.streamlit.app` URL and test a prediction.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 4. Troubleshooting

- **Model version error/warning:** pin compatible versions in `requirements.txt` matching the versions used in Colab, especially scikit-learn. Re-run the notebook and re-download the model if needed.
- **File not found:** confirm `customer_churn_model.joblib` is beside `app.py` in GitHub.
- **Missing package:** add the missing third-party package to `requirements.txt`, commit the change, and wait for redeployment.
- **Unexpected prediction:** confirm the uploaded model came from this notebook. Predictions are educational estimates, not guarantees about real customers.
