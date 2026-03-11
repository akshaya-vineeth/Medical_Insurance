# 1. Load scaler.pkl and model.pkl files
# 2. Create Function to predict 

import pickle
import numpy as np


class Insurance_Prediction:
    def __init__(self):
        with open('artifacts/scaler.pkl','rb') as f:
            self.scaler = pickle.load(f)
        with open('artifacts/model.pkl','rb') as f:
            self.model = pickle.load(f)


    
    def predict(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        inputt = np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        scaled_input = self.scaler.transform(inputt)
        predicted_value = self.model.predict(scaled_input)
        return predicted_value[0]

    def prediction(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        return self.predict(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    

        