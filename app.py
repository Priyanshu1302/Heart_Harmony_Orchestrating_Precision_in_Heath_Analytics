from flask import Flask, render_templates, request
import pickle
import numpy as np


filename = 'logreg.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_templates('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        smoking = int(request.form['glucose'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[smoking, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_templates('result.html', prediction=my_prediction)

if __name__ == 'main.html':
	app.run(debug=True)