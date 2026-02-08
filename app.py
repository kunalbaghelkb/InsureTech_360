import os
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from src.pipelines.prediction_pipeline import FraudData, FraudPredictor, DamagePredictor
from src.components.rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)

# Global Initialization
try:
    print("Initializing RAG Chatbot...")
    rag_engine = RAGEngine()
    print("Chatbot Ready!")
except Exception as e:
    print(f"Chatbot Failed to Load: {e}")
    rag_engine = None

# Routes
@app.route('/')
def home():
    return render_template('index.html', current_year=datetime.now().year)

@app.route('/predict_fraud', methods=['GET', 'POST'])
def predict_fraud():
    if request.method == 'GET':
        return render_template('index.html', current_year=datetime.now().year)
    
    try:
        # Get Data from HTML Form (Keep name attributes in HTML form the same as in FraudData class)
        data = FraudData(
            # Numerical Features
            months_as_customer = int(request.form.get('months_as_customer')),
            age = int(request.form.get('age')),
            policy_deductable = int(request.form.get('policy_deductable')),
            policy_annual_premium = float(request.form.get('policy_annual_premium')),
            umbrella_limit = int(request.form.get('umbrella_limit')),
            capital_gains = int(request.form.get('capital_gains')),
            capital_loss = int(request.form.get('capital_loss')),
            incident_hour_of_the_day = int(request.form.get('incident_hour_of_the_day')),
            number_of_vehicles_involved = int(request.form.get('number_of_vehicles_involved')),
            bodily_injuries = int(request.form.get('bodily_injuries')),
            witnesses = int(request.form.get('witnesses')),
            total_claim_amount = int(request.form.get('total_claim_amount')),
            
            # Categorical Features
            sex = request.form.get('sex'),
            marital_status = request.form.get('marital_status'),
            fault = request.form.get('fault'),
            accident_area = request.form.get('accident_area'),
            police_report_filed = request.form.get('police_report_filed'),
            witness_present = request.form.get('witness_present'),
            vehicle_category = request.form.get('vehicle_category')
        )
        
        # Convert to DataFrame
        pred_df = data.get_data_as_df()
        print(f"User Input:\n{pred_df}")

        # Predict
        fraud_predictor = FraudPredictor()
        result = fraud_predictor.predict(pred_df)
        
        # Return Result
        status = "FRAUDULENT" if result == 1 else "GENUINE"
        return render_template('index.html', fraud_result=status, current_year=datetime.now().year)

    except Exception as e:
        print(f"Error in Fraud Prediction: {e}")
        return render_template('index.html', fraud_result=f"Error: {e}", current_year=datetime.now().year)

@app.route('/predict_damage', methods=['POST'])
def predict_damage():
    try:
        if 'file' not in request.files:
            return render_template('index.html', damage_result="No file uploaded", current_year=datetime.now().year)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', damage_result="No file selected", current_year=datetime.now().year)

        # Save Image Temporarily
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
            
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)

        # Predict
        damage_predictor = DamagePredictor()
        result = damage_predictor.predict(file_path)

        return render_template('index.html', damage_result=result, uploaded_image=file_path, current_year=datetime.now().year)

    except Exception as e:
        print(f"Error in Damage Prediction: {e}")
        return render_template('index.html', damage_result=f"Error: {e}", current_year=datetime.now().year)

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