import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Charger le modèle
model=pickle.load(open('ckd_model.pkl','rb'))
scalar=pickle.load(open('data_scaling.pkl','rb'))
# page principale
@app.route('/')
def home():
    return render_template('home.html')
# page de prédiction 
@app.route('/predict_api',methods=['POST'])
def predict_api():
    try:
    
        data = request.json['data']
        print(data)
        print(np.array(list(data.values())))
        print((np.array(list(data.values())).reshape(1,-1)))
        output=model.predict(np.array(list(data.values())).reshape(1,-1))
        print(output)
        if (output[0]== 0):
            print('La personne ne a pas le CKD')
        else:
            print('La personne  a le CKD')
        return jsonify(int(output[0]))

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    outPut=model.predict(np.array((data)).reshape(1,-1))
    #final_input=model.transform(np.array(data).reshape(1,-1))
    print(outPut)
    #output=Classmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="L'état de patient(e): {}".format(outPut))



if __name__=="__main__":
    app.run(debug=True)




