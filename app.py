from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("customer_churn_model.joblib")

st.title("📊 Customer Churn Prediction")
st.write(
    "Estimate whether a telecom customer is at risk of leaving based on their "
    "service, contract, tenure, and billing details. This is a student ML demo, "
    "not a definitive decision about a real customer."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "customer_churn_model.joblib was not found. Train the model in Colab, "
            "download the file, and place it beside app.py."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.subheader("Enter customer details")
st.caption("Choose realistic, internally consistent values for the customer.")

with st.form("customer_details"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has a partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col2:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines_options = ["No phone service"] if phone_service == "No" else ["No", "Yes"]
        multiple_lines = st.selectbox("Multiple phone lines", multiple_lines_options)
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        if internet_service == "No":
            online_security = st.selectbox("Online security", ["No internet service"], disabled=True)
            online_backup = st.selectbox("Online backup", ["No internet service"], disabled=True)
            device_protection = st.selectbox("Device protection", ["No internet service"], disabled=True)
            tech_support = st.selectbox("Tech support", ["No internet service"], disabled=True)
        else:
            online_security = st.selectbox("Online security", ["No", "Yes"])
            online_backup = st.selectbox("Online backup", ["No", "Yes"])
            device_protection = st.selectbox("Device protection", ["No", "Yes"])
            tech_support = st.selectbox("Tech support", ["No", "Yes"])

    with col3:
        if internet_service == "No":
            streaming_tv = st.selectbox("Streaming TV", ["No internet service"], disabled=True)
            streaming_movies = st.selectbox("Streaming movies", ["No internet service"], disabled=True)
        else:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total charges to date ($)", min_value=0.0, max_value=20000.0,
            value=float(tenure) * float(monthly_charges), step=10.0,
            help="Use the customer's actual cumulative bill when available. The default is only an estimate based on tenure × monthly charges.",
        )

    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    add_on_services = [
        multiple_lines, online_security, online_backup, device_protection,
        tech_support, streaming_tv, streaming_movies,
    ]
    num_additional_services = sum(value == "Yes" for value in add_on_services)

    # Keep these raw column names aligned with X in the Colab notebook.
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "NumAdditionalServices": int(num_additional_services),
    }])

    try:
        prediction = int(model.predict(customer)[0])
        churn_probability = float(model.predict_proba(customer)[0][1])
    except Exception as exc:
        st.error(f"Prediction failed. Check that the model was trained using the supplied notebook. Details: {exc}")
        st.stop()

    st.divider()
    st.subheader("Prediction result")
    left, right = st.columns(2)
    with left:
        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
    with right:
        st.metric("Additional services selected", num_additional_services)

    if prediction == 1:
        st.warning("The model classifies this customer as likely to churn.")
        st.write("A business might review service quality, contract options, or support needs before deciding on any retention action.")
    else:
        st.success("The model classifies this customer as not likely to churn.")
        st.write("This prediction does not guarantee that the customer will stay.")

    with st.expander("Input summary"):
        st.dataframe(customer.T.rename(columns={0: "Selected value"}), use_container_width=True)

st.divider()
st.caption(
    "Model output is an estimate based on a historical sample dataset. It may be inaccurate "
    "for current customers or other telecom providers; evaluate fairness and business impact before use."
)
