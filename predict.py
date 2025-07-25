import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("best_model_pipeline.pkl")
feature_names = joblib.load("features.pkl")
model_for_importances = model.named_steps['rf']
# print(model.named_steps['rf'].feature_importances_)

st.set_page_config(page_title="Pregnancy Risk Predictor", layout="centered")
st.title("🤰 Pregnancy Risk Prediction")
st.markdown("Fill in the pregnancy details to predict **High Risk Pregnancy**.")

# Input form
with st.form("pregnancy_form"):
    age = st.number_input("Age", 15, 50, value=25)
    gravida = st.selectbox("Gravida", [1, 2, 3, 4], index=0)
    titi_tika = st.selectbox("TiTi Tika", [1, 2, 3], index=0)
    gestational_age = st.number_input("Gestational Age (weeks)", 1, 45, value=38)
    weight = st.number_input("Weight (kg)", 30, 150, value=60)
    # height_cm = st.number_input("Height (cm)", 100.0, 200.0, value=160.0)

    systolic_bp = st.number_input("Systolic Blood Pressure", 30, 200, value=100)
    diastolic_bp = st.number_input("Diastolic Blood Pressure", 30, 130, value=70)

    anemia = st.selectbox("Anemia", ["No", "Minimal","Medium"])
    jaundice = st.selectbox("Jaundice", ["No", "Minimal","Medium"])
    # fetal_position = st.selectbox("Fetal Position", ["Normal", "Abnormal"])
    # fetal_movement = st.selectbox("Fetal Movement", ["Normal", "Abnormal"])
    fetal_heartbeat = st.number_input("Fetal Heartbeat", 100, 200, value=130)
    urine_albumin = st.selectbox("Urine Albumin", ["No", "Minimal","Medium","Higher"])
    vdrl = st.selectbox("VDRL", ["Negative", "Positive"])
    hrsag = st.selectbox("HBsAG", ["Negative", "Positive"])

    submitted = st.form_submit_button("Predict")

# On submit
if submitted:
    def to_binary(val):
        # return 1 if val in ["Yes", "Positive", "Abnormal"] else 0
        if val in ["No","Negative","Normal"]:
            return 0
        elif val in ["Minimal","Abnormal"]:
            return 1
        elif val in ["Medium"]:
            return 2
        else:
            return 3

    input_data = pd.DataFrame([{
        "Age": age,
        "Gravida": gravida,
        "TiTi Tika": titi_tika,
        "Gestational Age": gestational_age,
        "Weight": weight,
        "Anemia": to_binary(anemia),
        "Jaundice": to_binary(jaundice),
        # "Fetal Position": to_binary(fetal_position),
        # "Fetal Movement": to_binary(fetal_movement),
        "Fetal Heartbeat": fetal_heartbeat,
        "Urine Albumin": to_binary(urine_albumin),
        "VDRL": to_binary(vdrl),
        "HRsAG": to_binary(hrsag),
        # "Height_cm": height_cm,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp,
        # "Fetal position": to_binary(fetal_position)
    }])

    # Drop columns not used in training
    # input_data = input_data.drop(columns=["Fetal Position"])

    prediction = model.predict(input_data)[0]
    probabability = model.predict_proba(input_data)[0][1]
    # feature_importance = model.best_estimator_.feature_importances_

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High Risk Pregnancy Detected")
        st.write(f"**Probability of High Risk Pregnancy:** {probabability:.2%}")
    else:
        st.success("✅ Not a High Risk Pregnancy")
        st.write(f"**Probability of High Risk Pregnancy:** {probabability:.2%}")

    def plot_importance(model_name):
         importances = model_name.feature_importances_
         sorted_idx = np.argsort(importances)[::-1]
         sorted_features = [feature_names[i] for i in sorted_idx]
         sorted_importances = importances[sorted_idx]

         fig, ax = plt.subplots()
         sns.barplot(x=sorted_importances, y=sorted_features, ax=ax)
         ax.set_title(f"Feature Importance")
         ax.set_xlabel("Importance")
         ax.set_ylabel("Feature")
         st.pyplot(fig)

         st.subheader("Feature Importance")
    # plot_importance(mother_model, "Mother")
    plot_importance(model_name=model_for_importances)


    
