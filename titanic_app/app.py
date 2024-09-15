from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open(r'C:\Users\Dell\titanic_app\titanic_model (1).pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

        features = [float(request.form['pclass']),
                    float(request.form['age']),
                    float(request.form['fare']),
                    float(request.form['sex']),
                    float(request.form['sibsp']),
                    float(request.form['parch']),
                    float(request.form['embarked']),
        ]


        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)


        output = 'Survived' if prediction[0] == 1 else 'Did not survive'

        return render_template('index.html', prediction_text=f'prediction: {output}')
if __name__ == "__main__":
    app.run(debug=True)