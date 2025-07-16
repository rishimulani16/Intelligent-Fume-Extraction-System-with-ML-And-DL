from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model and scaler
model = load_model('pressure_ID_Fan_inlet_DL(8).h5')
scaler = joblib.load("scaler(8).pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])

        # Create input array
        input_data = np.array([[f1, f2, f3, f4, f5]])

        # Scale the input data (all at once)
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        prediction_value = prediction[0][0] if prediction.shape == (1, 1) else prediction.tolist()

        return render_template('index.html', prediction=prediction_value)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
