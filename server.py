from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import shap

app = Flask(__name__)
CORS(app)

# -------------------------------------------------------
# LOAD MODEL + SYMPTOM COLUMNS
# -------------------------------------------------------
model = joblib.load("disease_prediction_best_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

# -------------------------------------------------------
# Preload SHAP Explainer
# -------------------------------------------------------
# Reverted to original setup as requested.
background = pd.DataFrame(np.zeros((1, len(symptom_columns))), columns=symptom_columns)
explainer = shap.Explainer(model, background)


# -------------------------------------------------------
# API ROUTE
# -------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    selected = data.get("symptoms", [])

    # -------------------------------------------------------
    # Create model input vector
    # -------------------------------------------------------
    input_vec = [1 if s in selected else 0 for s in symptom_columns]
    input_df = pd.DataFrame([input_vec], columns=symptom_columns)

    # -------------------------------------------------------
    # Prediction
    # -------------------------------------------------------
    prediction = model.predict(input_df)[0]

    # -------------------------------------------------------
    # Top 3 Probabilities
    # -------------------------------------------------------
    proba = model.predict_proba(input_df)[0]
    top_idx = np.argsort(proba)[::-1][:3]

    top3 = []
    for i in top_idx:
        top3.append({
            "disease": str(model.classes_[i]),
            "confidence": float(proba[i] * 100)
        })

    # -------------------------------------------------------
    # SHAP Explainability
    # (Handles all weird SHAP shapes same as your Streamlit app)
    # -------------------------------------------------------
    shap_result = explainer(input_df, check_additivity=False)
    vals = shap_result.values

    # Handle old & new SHAP formats
    if isinstance(vals, list):
        arr = vals[0]
        shap_arr = arr[0] if arr.ndim == 2 else np.array(arr).reshape(-1)
    else:
        vals = np.array(vals)
        if vals.ndim == 3:
            # Assumes prediction is the index of the class you want to explain
            predicted_class_index = np.where(model.classes_ == prediction)[0][0]
            shap_arr = vals[0, :, predicted_class_index]
        elif vals.ndim == 2:
            shap_arr = vals[0]
        else:
            shap_arr = vals.flatten()[:len(symptom_columns)]

    # Ensure same length
    if shap_arr.shape[0] != len(symptom_columns):
        fixed = np.zeros(len(symptom_columns))
        fixed[:min(len(shap_arr), len(fixed))] = shap_arr[:min(len(shap_arr), len(fixed))]
        shap_arr = fixed

    # Build SHAP dataframe
    shap_df = pd.DataFrame({
        "symptom": symptom_columns,
        "value": shap_arr
    })

    # Sort by absolute value
    shap_df = shap_df.reindex(shap_df["value"].abs().sort_values(ascending=False).index)
    top_shap = shap_df.head(10).to_dict(orient="records")

    # -------------------------------------------------------
    # RETURN JSON
    # -------------------------------------------------------
    return jsonify({
        "prediction": str(prediction),
        "top3": top3,
        "shap": top_shap
    })


# -------------------------------------------------------
# MAIN (Modified for Production)
# -------------------------------------------------------
if __name__ == "__main__":
    # In a production environment like Render, a WSGI server (like Gunicorn)
    # will run 'server:app' and ignore this conditional block.
    # We remove the explicit host/port to avoid conflicts on render.
    app.run()