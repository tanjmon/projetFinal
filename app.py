import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
model=pickle.load(open('ckd_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    print(data)
    features = np.array(data['features']).reshape(1, -1)
    #data=int(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(features)
    print(output[0])
    return jsonify(output[0])

#@app.route('/predict',methods=['POST'])
#def predict():
    #data=[float(x) for x in request.form.values()]
    #final_input=model.transform(np.array(data).reshape(1,-1))
    #print(final_input)
    #output=Classmodel.predict(final_input)[0]
    #return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     