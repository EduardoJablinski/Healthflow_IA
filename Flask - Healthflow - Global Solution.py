from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Carregar o modelo salvo
modelo = joblib.load('random_forest.joblib')

df1 = pd.read_csv('dataset1_rf.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predd():
    # Obter os valores dos sintomas da solicitação POST
    S1 = request.form['S1']
    S2 = request.form['S2']
    S3 = request.form['S3']
    S4 = request.form['S4']
    S5 = request.form['S5']
    S6 = request.form['S6']
    S7 = request.form['S7']
    S8 = request.form['S8']
    S9 = request.form['S9']
    S10 = request.form['S10']
    S11 = request.form['S11']
    S12 = request.form['S12']
    S13 = request.form['S13']
    S14 = request.form['S14']
    S15 = request.form['S15']
    S16 = request.form['S16']
    S17 = request.form['S17']

    psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
    
    a = np.array(df1["Sintoma"])
    b = np.array(df1["peso"])
    
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]
    
    psy = [psymptoms]
    pred2 = modelo.predict(psy)
    print("Doença: ", pred2[0])

    return render_template('result.html', doenca=pred2)

if __name__ == '__main__':
    app.run(debug=True)
