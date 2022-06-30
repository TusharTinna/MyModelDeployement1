import pandas as pd
import numpy as np
from flask import Flask, render_template,request
import pickle

app=Flask(__name__)
data=pd.read_csv("real_estate_price_size_year_view.csv")
pipe=pickle.load(open("House_Model.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sea = float(request.form.get('Sea'))
    size = float(request.form.get('size'))
    print(sea,size)

    input=pd.DataFrame([[size,sea]],columns=["size","sea view"])
    prediction=pipe.predict(input) # I have converted Lakhs into Crores. Since data set is old, so I have added 0.2(20lakhs to compansate inflation.
    return "As per your requirement the predicted house price is {} ".format(prediction)

if __name__=="__main__":
    app.run(debug=True,port=5000)