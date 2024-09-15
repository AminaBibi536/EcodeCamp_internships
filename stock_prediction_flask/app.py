from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model(r'C:\Users\Dell\stock_prediction_flask\stock_price_prediction_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    input_data = [float(x) for x in input_data.split(',')]
    input_data= np.array(input_data).reshape(1, -1, 1)
    prediction = model.predict(input_data)
    return render_template('index.html', prediction_text='Predicted Stock Price: ${:.2f}'.format(prediction[0][0]))


if __name__ == "__main__":
    app.run(debug=True)
