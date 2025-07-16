from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load models and scalers
model1 = load_model('pressure_ID_Fan_inlet_DL(1).h5')
scaler1 = joblib.load("scaler(1).pkl")

model2 = load_model('pressure_at_bag_house_DL(2).h5')
scaler2 = joblib.load("scaler(2).pkl")

model3 = load_model('pressure_at_gas_mixture_DL(3).h5')
scaler3 = joblib.load("scaler(3).pkl")

model4 = load_model('FDC_Outlet_Pressure_DL(4).h5')
scaler4 = joblib.load("scaler(4).pkl")

model5 = load_model('pressure_at_combustion_DL(5).h5')
scaler5 = joblib.load("scaler(5).pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all 23 inputs from form
        f = []
        for i in range(23):  # f0 to f22
            f.append(float(request.form[f'f{i}']))

        # Create input arrays using correct index offsets (0-based)
        input_data_1 = np.array([[f[15], f[17], f[20], f[21], f[22]]])  #f[16] output
        input_data_2 = np.array([[f[14], f[15], f[16], f[17], f[20], f[21], f[22]]]) #f[13] output
        input_data_3 = np.array([[f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[20], f[21], f[22]]])  #f[10] output
        input_data_4 = np.array([[f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[20], f[21], f[22]]])  #f[9] output
        input_data_5 = np.array([[f[0], f[1], f[2], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[20], f[21], f[22]]])  #f[3] output

        #actual outputs
        actual_data_1 = f[16]
        actual_data_2 = f[13]
        actual_data_3 = f[10]
        actual_data_4 = f[9]
        actual_data_5 = f[3]

        # Debug info
        print(f"input_data_1 shape: {input_data_1.shape}")
        print(f"scaler1 expects: {scaler1.n_features_in_}")

        # Scale inputs
        input_scaled_1 = scaler1.transform(input_data_1)
        input_scaled_2 = scaler2.transform(input_data_2)
        input_scaled_3 = scaler3.transform(input_data_3)
        input_scaled_4 = scaler4.transform(input_data_4)
        input_scaled_5 = scaler5.transform(input_data_5)

        # Predict
        prediction1 = model1.predict(input_scaled_1)
        prediction2 = model2.predict(input_scaled_2)
        prediction3 = model3.predict(input_scaled_3)
        prediction4 = model4.predict(input_scaled_4)
        prediction5 = model5.predict(input_scaled_5)

        #diff 
        diff_1 = abs(prediction1 - actual_data_1)
        diff_2 = abs(prediction2 - actual_data_2)
        diff_3 = abs(prediction3 - actual_data_3)
        diff_4 = abs(prediction4 - actual_data_4)
        diff_5 = abs(prediction5 - actual_data_5)

        # Extract result
        result1 = None
        result2 = None
        result3 = None
        result4 = None
        result5 = None

        # Extract values
        prediction_value1 = prediction1[0][0] if prediction1.shape == (1, 1) else prediction1.tolist()
        if diff_1 > 20:
            result1 = "problem"

        if diff_2 > 5:
            result2 = "problem"

        if diff_3 > 12:
            result3 = "problem"

        if diff_4 > 20:
            result4 = "problem"

        if diff_5 > 1:
            result5 = "problem"

        prediction_value2 = prediction2[0][0] if prediction2.shape == (1, 1) else prediction2.tolist()
        prediction_value3 = prediction3[0][0] if prediction3.shape == (1, 1) else prediction3.tolist()
        prediction_value4 = prediction4[0][0] if prediction4.shape == (1, 1) else prediction4.tolist()
        prediction_value5 = prediction5[0][0] if prediction5.shape == (1, 1) else prediction5.tolist()

        return render_template(
            'index.html',
            prediction1=prediction_value1,
            prediction2=prediction_value2,
            prediction3=prediction_value3,
            prediction4=prediction_value4,
            prediction5=prediction_value5,

            actual_data_1 = actual_data_1,
            actual_data_2 = actual_data_2,
            actual_data_3 = actual_data_3,
            actual_data_4 = actual_data_4,
            actual_data_5 = actual_data_5,

            result1 = result1,
            result2 = result2,
            result3 = result3,
            result4 = result4,
            result5 = result5
        )


    except Exception as e:
        print("Full Error:", e)
        return render_template('index.html', prediction_error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

