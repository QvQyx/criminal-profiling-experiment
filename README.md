# Criminal Profiling — A Python Experiment

Inspired by my research into family disadvantage and teenage crime, I wanted to explore whether machine learning could identify patterns in criminal behaviour.

I built a Random Forest classifier using 500 simulated records and a Streamlit interface for exploring its predictions. Through the project, I realised that the model was learning the rules I had created—not patterns verified by real-world evidence.

Working with simulated data helped me realise that a dataset can follow clear rules without reflecting real-world behaviour. This taught me the importance of understanding how a dataset is created, what assumptions it contains, and what conclusions it can support.

**Python · NumPy · pandas · scikit-learn · Streamlit**

[Try the live demo](https://qvqyx-criminal-profiling.streamlit.app/)

## Try it locally

```sh
pip install -r requirements.txt
streamlit run app.py
```

The model trains automatically when the app starts. No API keys are needed.

## How it works

The dataset contains 500 fictional records with age, income, and previous crime count. Labels follow a preset count rule: 0–1 = No or Minor Crime, 2–3 = Theft, and 4+ = Violent Crime. These are artificial assumptions, not established relationships. Age and income do not determine the labels.

A seeded Random Forest learns from 400 rows and is evaluated on 100 held-out rows. Both the model and the original rule achieve 100% accuracy on this split; the majority baseline scores 53%. This demonstrates rule recovery, not real-world crime prediction.

`app.py` is the interface; `pipeline.py` contains the shared training code. Run `python pipeline.py train` to save the evaluation results locally.
