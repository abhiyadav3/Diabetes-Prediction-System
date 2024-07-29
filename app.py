from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)


scaler = joblib.load(r'Model/scaler.pkl')
svm = joblib.load(r'Model/svm_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_as_numpy_array)
    prediction = svm.predict(std_data)
    result = 'The person is diabetic' if prediction[0] else 'The person is not diabetic'
    return render_template('index.html', prediction_text=result)
