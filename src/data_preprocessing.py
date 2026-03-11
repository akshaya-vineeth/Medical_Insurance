#   1 . Loading Raw Data
#   2.  Indentifying x and y (input and output)
#   3.  Split Data Into training and testing



import pandas as pd
from sklearn.model_selection import train_test_split

def load_split_data():
    data = pd.read_csv("D:\ML-DL PROJECTS\ML\LinearRegression\MedicalInsurance\Medical_Insurance\data\\raw\insurance.csv")



    x = data[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y = data['Annual_Premium_Thousands']

    x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train , x_test , y_train , y_test




