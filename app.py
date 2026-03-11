import os
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_cors import CORS
from src.pipelines.prediction_pipeline import FraudData, FraudPredictor, DamagePredictor
from src.components.rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)

# Inject current_year into every template automatically
@app.context_processor
def inject_globals():
    return {"current_year": datetime.now().year}

# Global Initialization
try:
    print("Initializing RAG Chatbot...")
    rag_engine = RAGEngine()
    print("Chatbot Ready!")
except Exception as e:
    print(f"Chatbot Failed to Load: {e}")
    rag_engine = None

# Routes
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve images saved to /tmp/uploads (writable on Hugging Face Spaces)."""
    return send_from_directory('/tmp/uploads', filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_fraud', methods=['GET', 'POST'])
def predict_fraud():
    if request.method == 'GET':
        return render_template('index.html')
    
    try:
        # Read all 31 real CSV columns from the form — no hardcoded defaults
        data = FraudData(
            # Numeric
            age                    = int(request.form.get('age')),
            week_of_month          = int(request.form.get('week_of_month')),
            week_of_month_claimed  = int(request.form.get('week_of_month_claimed')),
            rep_number             = int(request.form.get('rep_number')),
            deductible             = int(request.form.get('deductible')),
            driver_rating          = int(request.form.get('driver_rating')),
            year                   = int(request.form.get('year')),
            # Categorical
            month                  = request.form.get('month'),
            day_of_week            = request.form.get('day_of_week'),
            make                   = request.form.get('make'),
            accident_area          = request.form.get('accident_area'),
            day_of_week_claimed    = request.form.get('day_of_week_claimed'),
            month_claimed          = request.form.get('month_claimed'),
            sex                    = request.form.get('sex'),
            marital_status         = request.form.get('marital_status'),
            fault                  = request.form.get('fault'),
            policy_type            = request.form.get('policy_type'),
            vehicle_category       = request.form.get('vehicle_category'),
            vehicle_price          = request.form.get('vehicle_price'),
            days_policy_accident   = request.form.get('days_policy_accident'),
            days_policy_claim      = request.form.get('days_policy_claim'),
            past_number_of_claims  = request.form.get('past_number_of_claims'),
            age_of_vehicle         = request.form.get('age_of_vehicle'),
            age_of_policy_holder   = request.form.get('age_of_policy_holder'),
            police_report_filed    = request.form.get('police_report_filed'),
            witness_present        = request.form.get('witness_present'),
            agent_type             = request.form.get('agent_type'),
            number_of_suppliments  = request.form.get('number_of_suppliments'),
            address_change_claim   = request.form.get('address_change_claim'),
            number_of_cars         = request.form.get('number_of_cars'),
            base_policy            = request.form.get('base_policy'),
        )

        
        # Convert to DataFrame
        pred_df = data.get_data_as_df()
        print(f"User Input:\n{pred_df}")

        # Predict
        fraud_predictor = FraudPredictor()
        result = fraud_predictor.predict(pred_df)
        
        # Return Result
        status = "FRAUDULENT" if result == 1 else "GENUINE"
        return render_template('index.html', fraud_result=status)

    except Exception as e:
        print(f"Error in Fraud Prediction: {e}")
        return render_template('index.html', fraud_result=f"Error: {e}")

@app.route('/predict_damage', methods=['POST'])
def predict_damage():
    try:
        if 'file' not in request.files:
            return render_template('index.html', damage_result="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', damage_result="No file selected")

        # Save Image Temporarily to /tmp (writable on Hugging Face Spaces)
        upload_dir = '/tmp/uploads'
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Predict
        damage_predictor = DamagePredictor()
        result = damage_predictor.predict(file_path)

        # Pass a URL the browser can fetch (served by /uploads/ route)
        image_url = f"/uploads/{file.filename}"
        return render_template('index.html', damage_result=result, uploaded_image=image_url)

    except Exception as e:
        print(f"Error in Damage Prediction: {e}")
        return render_template('index.html', damage_result=f"Error: {e}")

@app.route('/ask_bot', methods=['POST'])
def ask_bot():
    try:
        # AJAX request will be done from Frontend
        data = request.json
        user_query = data.get('query')
        
        if not rag_engine:
            return jsonify({'answer': "Server Error: Bot is offline."})
        
        # Get Response
        response = rag_engine.get_response(user_query)
        
        # If LCEL object returns then convert it into string
        if not isinstance(response, str):
            response = str(response)

        return jsonify({'answer': response})

    except Exception as e:
        return jsonify({'answer': f"Error: {e}"})

if __name__ == "__main__":
    # Host 0.0.0.0 makes it accessible on network
    app.run(host="0.0.0.0", port=5050, debug=True)