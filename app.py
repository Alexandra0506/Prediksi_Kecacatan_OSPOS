# import streamlit as st
# import joblib
# import pandas as pd
# from scipy.sparse import hstack, csr_matrix

# # =========================================================
# # LOAD MODEL DAN PREPROCESSING OBJECT
# # =========================================================

# model = joblib.load("best_defect_prediction_model.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")
# scaler = joblib.load("numeric_scaler.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

# # =========================================================
# # STREAMLIT UI
# # =========================================================

# st.title("Prediksi Defect Modul OSPOS")
# st.write("Aplikasi ini digunakan untuk memprediksi tingkat risiko defect pada modul Point of Sale.")

# st.subheader("Input Source Code")

# source_code = st.text_area(
#     "Masukkan source code modul:",
#     height=300
# )

# st.subheader("Input Software Metrics")

# loc = st.number_input("LOC (Lines of Code)", min_value=0, value=0)
# cc = st.number_input("CC (Cyclomatic Complexity)", min_value=0, value=0)
# jml_karakter = st.number_input("Jumlah Karakter", min_value=0, value=0)
# jml_token = st.number_input("Jumlah Token", min_value=0, value=0)
# cbo = st.number_input("CBO", min_value=0, value=0)

# # =========================================================
# # PREDIKSI
# # =========================================================

# if st.button("Prediksi Risk Level"):

#     if source_code.strip() == "":
#         st.warning("Source code belum diisi.")
#     else:
#         # TF-IDF source code
#         X_text = vectorizer.transform([source_code])

#         # Numeric features
#         numeric_data = pd.DataFrame([{
#             "LOC": loc,
#             "CC": cc,
#             "Jml_Karakter": jml_karakter,
#             "Jml_Token": jml_token,
#             "CBO": cbo
#         }])

#         X_numeric_scaled = scaler.transform(numeric_data)
#         X_numeric_scaled = csr_matrix(X_numeric_scaled)

#         # Feature fusion
#         X_final = hstack([X_text, X_numeric_scaled])

#         # Prediction
#         prediction = model.predict(X_final)
#         risk_level = label_encoder.inverse_transform(prediction)[0]

#         st.subheader("Hasil Prediksi")
#         st.success(f"Risk Level: {risk_level}")

#         if risk_level == "Low Risk":
#             st.info("Modul memiliki risiko defect rendah.")
#         elif risk_level == "Medium Risk":
#             st.warning("Modul memiliki risiko defect sedang.")
#         elif risk_level == "High Risk":
#             st.error("Modul memiliki risiko defect tinggi dan perlu diprioritaskan untuk pengujian.")

import re
import joblib
import streamlit as st
import pandas as pd



# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Defect Prediction OSPOS",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #FFFFFF;
}

.title-box {
    background: linear-gradient(90deg, #1e3a8a, #2563eb);
    padding: 25px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.result-card {
    padding: 25px;
    border-radius: 18px;
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.05);
    text-align: center;
}

.metric-card {
    padding: 18px;
    border-radius: 14px;
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    color: #111827;
}

/* Text Area */
textarea {
    color: #111827 !important;
    background-color: #f9fafb !important;
    border: 1px solid #d1d5db !important;
}

/* Input Label */
label {
    color: #111827 !important;
    font-weight: 600;
}

/* Button styling */
div[data-testid="stButton"] button p {
    color: #ffffff !important;
}

div[data-testid="stButton"] button {
    background-color: #2563eb !important;
    border: none !important;
}

/* Headers and Caption */
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] span {
    color: #000000 !important;
}

div[data-testid="stMarkdownContainer"] .title-box h1,
div[data-testid="stMarkdownContainer"] .title-box p {
    color: #ffffff !important;
}

/* Metrics (LOC, CC, dll) */
div[data-testid="stMetricValue"] {
    color: #000000 !important;
}

div[data-testid="stMetricLabel"] p {
    font-size: 1.25rem !important;
    font-weight: bold !important;
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_objects():
    model = joblib.load("best_defect_prediction_model_no_smote.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_objects()

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_numeric_features(source_code):
    loc = len(source_code.splitlines())

    decision_keywords = re.findall(
        r'\b(if|while|for|foreach|case|catch)\b',
        source_code
    )

    boolean_ops = re.findall(
        r'&&|\|\|',
        source_code
    )

    ternary_ops = re.findall(
        r'\?',
        source_code
    )

    cc = 1 + len(decision_keywords) + len(boolean_ops) + len(ternary_ops)

    jml_karakter = len(source_code)
    jml_token = len(re.findall(r'\w+', source_code))

    return loc, cc, jml_karakter, jml_token

def predict_risk(source_code):
    loc, cc, jml_karakter, jml_token = extract_numeric_features(source_code)

    X_input = pd.DataFrame([{
        "Source_Code": source_code,
        "LOC": loc,
        "CC": cc,
        "Jml_Karakter": jml_karakter,
        "Jml_Token": jml_token
    }])

    prediction = model.predict(X_input)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X_input)[0]
        probability_df = pd.DataFrame({
            "Risk Level": label_encoder.classes_,
            "Probability": probability
        })
    else:
        probability_df = None

    return prediction_label, probability_df, loc, cc, jml_karakter, jml_token

# =========================================================
# UI
# =========================================================

st.markdown("""
<div class="title-box">
    <h1 style="color: white !important;">🧠 Software Defect Prediction</h1>
    <p style="color: white !important;">Prediksi tingkat risiko defect pada modul Point of Sale berbasis Open Source POS.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    st.subheader("📄 Input Source Code")
    source_code = st.text_area(
        "Masukkan source code PHP",
        height=420,
        placeholder="Paste source code PHP di sini..."
    )

with right:
    st.subheader("ℹ️ Informasi Model")
    st.markdown("""
    <div class="metric-card">
    <b>Model:</b> XGBoost / Model terbaik<br>
    <b>Fitur teks:</b> TF-IDF<br>
    <b>Fitur numerik:</b> LOC, CC, Jumlah Karakter, Jumlah Token<br>
    <b>Metode:</b> Multimodal Feature Fusion
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    predict_button = st.button("🔍 Prediksi Risk Level", type="primary", use_container_width=True)

# =========================================================
# RESULT
# =========================================================

if predict_button:
    if source_code.strip() == "":
        st.warning("Masukkan source code terlebih dahulu.")
    else:
        prediction_label, probability_df, loc, cc, jml_karakter, jml_token = predict_risk(source_code)

        st.write("---")
        st.subheader("📊 Hasil Prediksi")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("LOC", loc)
        col2.metric("CC", cc)
        col3.metric("Jumlah Karakter", jml_karakter)
        col4.metric("Jumlah Token", jml_token)

        st.write("")

        if prediction_label == "High Risk":
            color = "#dc2626"
            emoji = "🔴"
        elif prediction_label == "Medium Risk":
            color = "#f59e0b"
            emoji = "🟡"
        else:
            color = "#16a34a"
            emoji = "🟢"

        st.markdown(f"""
        <div class="result-card">
            <h2 style="color: black;">{emoji} Predicted Risk Level</h2>
            <h1 style="color:{color};">{prediction_label}</h1>
        </div>
        """, unsafe_allow_html=True)

        if probability_df is not None:
            st.write("")
            st.subheader("📈 Probabilitas Prediksi")
            st.dataframe(probability_df, use_container_width=True)

            st.bar_chart(
                probability_df.set_index("Risk Level")
            )

# =========================================================
# FOOTER
# =========================================================

st.write("---")
st.caption("Sistem prediksi defect menggunakan TF-IDF, software metrics, dan feature fusion.")