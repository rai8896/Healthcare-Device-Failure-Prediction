import streamlit as st
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------------------
# Load Model
# ----------------------------

model = joblib.load("failure_prediction_model.pkl")
importance = pd.read_csv("feature_importance.csv")

with open("kmf_model.pkl", "rb") as f:
    kmf = pickle.load(f)

# ----------------------------
# UI CONFIG
# ----------------------------

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

st.title(" Predictive Maintenance System (NASA CMAPSS)")
st.write("AI-powered Failure Prediction + Survival Analysis")

# ----------------------------
# INPUT SECTION
# ----------------------------

st.sidebar.header("Device Sensor Input")

features = [
    "engine_id","cycle","setting1","setting2",
    "sensor_2","sensor_3","sensor_4","sensor_6","sensor_7",
    "sensor_8","sensor_9","sensor_11","sensor_12","sensor_13",
    "sensor_14","sensor_15","sensor_17","sensor_20","sensor_21"
]

input_data = {}

for f in features:
    input_data[f] = st.sidebar.number_input(f, value=500.0)

input_df = pd.DataFrame([input_data])

# ----------------------------
# PREDICTION
# ----------------------------

if st.button(" Predict Failure"):

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    # ----------------------------
    # RISK LEVEL
    # ----------------------------

    if prob < 0.2:
        risk = "🟢 LOW RISK (Healthy)"
        color = "green"
    elif prob < 0.5:
        risk = "🟡 MEDIUM RISK"
        color = "orange"
    else:
        risk = "🔴 HIGH RISK"
        color = "red"

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Failure Probability", f"{prob*100:.2f}%")
        st.markdown(f"### {risk}")

    # ----------------------------
    # GAUGE METER (IMPORTANT)
    # ----------------------------

    with col2:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Failure Risk Gauge"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 20], 'color': "green"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
            }
        ))

        st.plotly_chart(fig)

    # ----------------------------
    # RESULT
    # ----------------------------

    if prob < 0.2:
        st.success(" LOW RISK - Machine is Healthy")
    elif prob < 0.5:
        st.warning(" MEDIUM RISK - Monitor Required")
    else:
     st.error(" HIGH RISK - Immediate Attention Needed")

# ----------------------------
# FEATURE IMPORTANCE
# ----------------------------

st.subheader("📊 Top Failure Contributing Sensors")

fig2, ax = plt.subplots()
top_features = importance.sort_values("Importance", ascending=False).head(10)

ax.barh(top_features["Feature"], top_features["Importance"])
ax.invert_yaxis()

st.pyplot(fig2)

# ----------------------------
# SURVIVAL ANALYSIS
# ----------------------------

st.subheader(" Device Survival Curve (Kaplan-Meier)")

fig3, ax3 = plt.subplots()
kmf.plot_survival_function(ax=ax3)

plt.xlabel("Cycles")
plt.ylabel("Survival Probability")

st.pyplot(fig3)