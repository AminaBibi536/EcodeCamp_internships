from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open(r'C:\Users\Dell\breast_cancer_app\breast_cancer_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        if request.method == 'POST' :

           features = [float(request.form['feature1']),
                       float(request.form['feature2']),
                       float(request.form['feature3']),
                       float(request.form['feature4']),
                       float(request.form['feature5']),
                       float(request.form['feature6']),
                       float(request.form['feature7']),
                       float(request.form['feature8']),
                       float(request.form['feature9']),
                       float(request.form['feature10']),
                       float(request.form['feature11']),
                       float(request.form['feature12']),
                       float(request.form['feature13']),
                       float(request.form['feature14']),
                       float(request.form['feature15']),
                       float(request.form['feature16']),
                       float(request.form['feature17']),
                       float(request.form['feature18']),
                       float(request.form['feature19']),
                       float(request.form['feature20']),
                       float(request.form['feature21']),
                       float(request.form['feature22']),
                       float(request.form['feature23']),
                       float(request.form['feature24']),
                       float(request.form['feature25']),
                       float(request.form['feature26']),
                       float(request.form['feature27']),
                       float(request.form['feature28']),
                       float(request.form['feature29']),
                       float(request.form['feature30']),
                       
                    ]

        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)

        result = 'Cancer Detected' if prediction[0] == 1 else ' No Cancer'

        return render_template('result.html', prediction=result)
if __name__ == "__main__":
    app.run(debug=True)
