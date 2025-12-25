from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import shap
import boto3
from botocore.exceptions import ClientError
import os
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "Authorization"],
    "methods": ["GET", "POST", "OPTIONS"]
}})
# -------------------------------------------------------
# LOAD MODEL + SYMPTOM COLUMNS
# -------------------------------------------------------
model = joblib.load("disease_prediction_best_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

# -------------------------------------------------------
# AWS S3 CONFIGURATION (from environment variables)
# -------------------------------------------------------
s3_client = boto3.client(
    's3',
    region_name=os.environ.get('AWS_REGION', 'ap-south-1'),
    aws_access_key_id='AKIAU7YUYBPKDLO7AVPT',
    aws_secret_access_key='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'
)
S3_BUCKET = 'healthapp-ai'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------------------------------
# Preload SHAP Explainer
# -------------------------------------------------------
background = pd.DataFrame(np.zeros((1, len(symptom_columns))), columns=symptom_columns)
explainer = shap.Explainer(model, background)


# -------------------------------------------------------
# DISEASE PREDICTION API ROUTE
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
# S3 UPLOAD ENDPOINT
# -------------------------------------------------------
@app.route("/s3/upload", methods=["POST"])
def s3_upload():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files['file']
        user_id = request.form.get('userId')
        file_name = request.form.get('fileName')
        
        if not user_id or not file_name:
            return jsonify({"success": False, "error": "Missing userId or fileName"}), 400
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Only PNG, JPG, JPEG allowed"}), 400
        
        # Clean the filename - remove special characters and spaces
        # Keep only alphanumeric, dots, hyphens, and underscores
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', file_name)
        
        # Ensure proper extension
        if '.' not in clean_name:
            # Get extension from original file
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpg'
            clean_name = f"{clean_name}.{ext}"
        
        s3_key = f"medical-reports/{user_id}/{clean_name}"
        
        print(f"Uploading file: {clean_name} to S3 key: {s3_key}")  # Debug log
        
        # Read file content
        file_content = file.read()
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType=file.content_type or 'image/jpeg',
            ACL='private'
        )
        
        # CHANGE THIS: Generate a presigned URL instead of a static one
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600
        )
        
        return jsonify({
            "success": True,
            "url": presigned_url,
            "key": s3_key,
            "fileName": clean_name
        })
        
    except ClientError as e:
        error_msg = str(e)
        print(f"AWS ClientError: {error_msg}")  # Debug log
        return jsonify({"success": False, "error": f"AWS Error: {error_msg}"}), 500
    except Exception as e:
        error_msg = str(e)
        print(f"General Error: {error_msg}")  # Debug log
        return jsonify({"success": False, "error": error_msg}), 500


# -------------------------------------------------------
# S3 GET SIGNED URL ENDPOINT
# -------------------------------------------------------
@app.route("/s3/get-signed-url", methods=["POST"])
def s3_get_signed_url():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        file_name = data.get('fileName')
        expires_in = data.get('expiresIn', 3600)  # Default 1 hour
        
        if not user_id or not file_name:
            return jsonify({"success": False, "error": "Missing userId or fileName"}), 400
        
        s3_key = f"medical-reports/{user_id}/{file_name}"
        
        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': s3_key
            },
            ExpiresIn=expires_in
        )
        
        return jsonify({
            "success": True,
            "url": url
        })
        
    except ClientError as e:
        return jsonify({"success": False, "error": f"AWS Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------------------------------------
# S3 DELETE ENDPOINT
# -------------------------------------------------------
@app.route("/s3/delete", methods=["POST"])
def s3_delete():
    try:
        data = request.get_json()
        user_id = data.get('userId')
        file_name = data.get('fileName')
        
        if not user_id or not file_name:
            return jsonify({"success": False, "error": "Missing userId or fileName"}), 400
        
        s3_key = f"medical-reports/{user_id}/{file_name}"
        
        # Delete from S3
        s3_client.delete_object(
            Bucket=S3_BUCKET,
            Key=s3_key
        )
        
        return jsonify({
            "success": True,
            "message": "File deleted successfully"
        })
        
    except ClientError as e:
        return jsonify({"success": False, "error": f"AWS Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -------------------------------------------------------
# HEALTH CHECK ENDPOINT
# -------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "model": "loaded",
            "s3": "configured"
        }
    })


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    app.run()