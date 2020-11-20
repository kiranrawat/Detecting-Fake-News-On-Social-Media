#!flask/bin/python
from flask import Flask, request, render_template, url_for
from flask_bootstrap import Bootstrap
import pandas as pd
import pickle

#ML Packages
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')
        
@app.route('/predict', methods=['POST'])
def predict():
    model_path="models/lr_final_model.pkl"
    transformer_path="models/transformer.pkl"
    # load the model and feature transformer with pickle
    loaded_model = pickle.load(open(model_path, 'rb'))
    loaded_transformer = pickle.load(open(transformer_path, 'rb'))
    # Collect the input and predict the outcome
    if request.method == 'POST':
        namequery = request.form['namequery']
        data = [namequery]
        test_features = loaded_transformer.transform(data)
        my_prediction = loaded_model.predict(test_features)
    return render_template('results.html', prediction=my_prediction, name = namequery.upper())

if __name__ == '__main__':
    app.run(debug=True)