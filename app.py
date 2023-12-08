from flask import Flask, render_template, jsonify,request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method =="POST":
        make = request.form.get('make')
        Model = request.form.get('model')
        engineCylinders = request.form.get('engineCylinders')
        year=request.form.get('year')
        df = pd.read_json('new.json')
        model_encode= df['Model_encode'][df['Model'] == Model].values[0]
        make_encode= df['Make_encode'][df['Make'] == make].values[0]

        print(make, Model,year, engineCylinders)

        with open('model.pkl', 'rb') as mod:
            mlmodel=pickle.load(mod)

        predit = mlmodel.predict([[make_encode,model_encode,float(year),float(engineCylinders)]])

        return render_template('predicted.html',predicted_value=predit[0])
    else:
        return render_template('predict.html')

if __name__=='__main__':
    app.run(host="0.0.0.0",port=66)