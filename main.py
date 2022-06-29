import pandas as pd
import numpy as np
from flask import Flask, render_template,request
import pickle

app=Flask(__name__)
pipe=pickle.load(open("House_Model.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sea = float(request.form.get('Sea'))
    sqft = float(request.form.get('total_sqft'))
    print(sea,sqft)
    data=[[sqft,sea]]
    input=pd.DataFrame(data,columns=["size","view"])
    prediction=pipe.predict(input) # I have converted Lakhs into Crores. Since data set is old, so I have added 0.2(20lakhs to compansate inflation.
    return "As per your requirement the predicted house price is {}".format(prediction)

if __name__=="__main__":
    app.run(debug=True,port=5000)
