import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import tempfile

# Load model and scaler
df = pd.read_csv("cleaned_early_disease.csv")
model = pickle.load(open("trained_disease_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🩺 Early Disease Prediction App")

# Choose input mode
mode = st.radio("Select Input Mode", ["Manual Input", "Upload CSV"])

if mode == "Manual Input":
    st.write("Enter the details below:")
    
    age = st.number_input("Age", 1, 120)
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (80–200)", 80, 200)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])
    height = st.number_input("Height (cm)", 100, 250)
    weight = st.number_input("Weight (kg)", 20, 200)
    sugar = st.number_input("Blood Sugar (mg/dL)", 70, 400)

    bmi = round(weight / ((height / 100) ** 2), 2)

    if st.button("Predict"):
        try:
            features = [[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]]

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            prediction_text = "✅ No Disease Detected" if prediction == 0 else "⚠️ Disease Detected"
            st.success(f"Prediction: {prediction_text}")

            # Generate PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Early Disease Prediction Report", ln=True, align='C')
            pdf.ln(10)

            fields = {
                "Age": age,
                "Sex": sex_label,
                "Chest Pain Type": cp,
                "Resting BP": trestbps,
                "Cholesterol": chol,
                "Fasting Blood Sugar > 120": fbs,
                "Resting ECG": restecg,
                "Max Heart Rate": thalach,
                "Exercise Induced Angina": exang,
                "Oldpeak": oldpeak,
                "Slope": slope,
                "Major Vessels": ca,
                "Thalassemia": thal,
                "Height (cm)": height,
                "Weight (kg)": weight,
                "BMI": bmi,
                "Sugar (mg/dL)": sugar,
                "Prediction": prediction_text
            }

            for key, value in fields.items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.output(tmp.name, 'F')
                st.download_button("📄 Download PDF Report", data=open(tmp.name, "rb"), file_name="prediction_report.pdf")

        except Exception as e:
            st.error(f"Error in prediction: {e}")

else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)

            # Ensure correct order and columns
            expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

            if not all(col in input_df.columns for col in expected_cols):
                st.error(f"CSV must contain columns: {expected_cols}")
            else:
                scaled = scaler.transform(input_df[expected_cols])
                predictions = model.predict(scaled)
                input_df["Prediction"] = ["No Disease" if p == 0 else "Disease" for p in predictions]

                st.success("Prediction completed successfully.")
                st.dataframe(input_df)

                # CSV download
                csv_download = input_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", csv_download, "disease_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Feature importance
st.subheader("🔍 Feature Importance")
importances = model.feature_importances_
feature_names = df.drop("target", axis=1).columns

fig, ax = plt.subplots()
ax.barh(feature_names, importances, color='skyblue')
ax.set_xlabel("Importance")
ax.set_title("Feature Importance (Random Forest)")
st.pyplot(fig)
