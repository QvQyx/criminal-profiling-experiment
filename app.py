"""A small, self-contained portfolio demo using fictional records."""
import pandas as pd
import streamlit as st
from pipeline import ROOT, FEATURES, TARGET, fit_model, rule_labels

st.set_page_config(page_title="Criminal Profiling — A Python Experiment", page_icon="🧪")
st.title("Criminal Profiling")
st.caption("A Python Experiment · 500 simulated records · Random Forest")
st.write("Inspired by my research into family disadvantage and teenage crime, I wanted to explore whether machine learning could identify patterns in criminal behaviour.")
st.info("This is a learning experiment using simulated data, not a tool for predicting real crime or assessing people.")

@st.cache_resource
def load_model():
    data = pd.read_csv(ROOT / "simulated_crime_data.csv")
    model, _, _ = fit_model(data)
    return model

model = load_model()
st.subheader("Try a fictional example")
st.write("Adjust the inputs to explore how the model responds.")
age = st.slider("Age", 18, 59, 30)
income = st.slider("Income (simulated units)", 0, 12000, 5000, step=100)
count = st.slider("Previous crime count (simulated)", 0, 10, 1)
record = pd.DataFrame({"Age": [age], "Income": [income], "Crime Count": [count]})
probabilities = model.predict_proba(record[FEATURES])[0]
st.subheader("Model output")
for category, score in zip(model.classes_, probabilities):
    st.progress(float(score), text=f"{category} — {score:.1%}")
st.caption("Scores apply only to this synthetic model; they are not real-world risk estimates.")
with st.expander("How does it work?"):
    st.write("The training labels follow a simple rule: count 0–1 → No or Minor Crime; 2–3 → Theft; 4+ → Violent Crime. These are arbitrary categories from the original experiment.")
    st.write("Rule category for this example:", str(rule_labels([count])[0]))
    st.write("Age and income do not determine the labels. The model learns from 400 records, with 100 held back for evaluation. The model and the original rule both achieve 100% accuracy on that test split: this shows rule recovery, not real-world validity.")
    st.write("Inputs outside the observed training range may produce less consistent results.")
st.subheader("What I learned")
st.write("Working with simulated data helped me realise that a dataset can follow clear rules without reflecting real-world behaviour. This taught me the importance of understanding how a dataset is created, what assumptions it contains, and what conclusions it can support.")
st.caption("Python · NumPy · pandas · scikit-learn · Streamlit")
