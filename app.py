from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the form data
        T = float(request.form['T'])
        TM = float(request.form['TM'])
        Tm = float(request.form['Tm'])
        SLP = float(request.form['SLP'])
        H = float(request.form['H'])
        VV = float(request.form['VV'])
        V = float(request.form['V'])
        VM = float(request.form['VM'])

        # Create input data as a numpy array
        input_data = np.array([[T, TM, Tm, SLP, H, VV, V, VM]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Pass the prediction and input data back to the template
        return render_template('index.html', prediction=prediction, T=T, TM=TM, Tm=Tm, SLP=SLP, H=H, VV=VV, V=V, VM=VM)

    # If GET request or initial load, render the template without prediction
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
