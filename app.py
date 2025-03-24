from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained Random Forest model and feature names
model = joblib.load('model_rf.joblib')  
features = joblib.load('features.joblib')

# Helper function to preprocess user input
def preprocess_input(form_data):
    input_data = []
    for feature in features:
        value = form_data.get(feature, 0)  # Default to 0 if missing
        try:
            input_data.append(float(value))  # Convert to float
        except ValueError:
            input_data.append(0)  # Handle invalid inputs
    return np.array(input_data).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        form_data = request.form
        user_input = preprocess_input(form_data)
        
        # Make prediction
        severity_prediction = model.predict(user_input)[0]
        
        # Severity Mapping
        severity_mapping = {0: 'Mild', 1: 'Moderate', 2: 'Severe', 3: 'Critical'}
        severity_result = severity_mapping.get(severity_prediction, 'Unknown')
        
        # Test Result Logic (Assuming severity > 0 means Positive)
        test_result = 'Positive' if severity_prediction > 0 else 'Negative'
        
        result = {'severity': severity_result, 'test_result': test_result}
        
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
