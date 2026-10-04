import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Base directory for resolving model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

FEATURE_COLUMNS = ['category', 'sellable_online', 'other_colors', 'depth', 'height', 'width']

def get_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    pour l'affichage sur html
    '''
    model = get_model()
    if model is None:
        return render_template('index.html', prediction_text="Erreur : model.pkl est introuvable. Veuillez exécuter le notebook et placer model.pkl dans le dossier de l'application.")

    try:
        # Extract features safely supporting both floats and ints
        category = float(request.form.get('category', 0))
        sellable_online = float(request.form.get('sellable_online', 0))
        other_colors = float(request.form.get('other_colors', 0))
        depth = float(request.form.get('depth', 0))
        height = float(request.form.get('height', 0))
        width = float(request.form.get('width', 0))

        features_df = pd.DataFrame(
            [[category, sellable_online, other_colors, depth, height, width]],
            columns=FEATURE_COLUMNS
        )
        print("Données d'entrée pour la prédiction :\n", features_df)

        prediction = model.predict(features_df)
        output = round(float(prediction[0]), 2)

        return render_template('index.html', prediction_text=f'Furniture prediction price is : $ {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Erreur lors du calcul : {e}')

if __name__ == "__main__":
    app.run(debug=True)