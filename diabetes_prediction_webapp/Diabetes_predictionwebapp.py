# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:07:11 2023

@author: shubh
"""
import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('D:/projects\diabetes pridiction sysytem/trained_model.sav','rb'))
# in pytohn we use forward slash to save in directory
# rb = reading the file in binary 
# load model is used to load model


# creating a function for prediction

def diabetes_prediction(input_data):
    # we are changing the array into numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped )
    print(prediction)
    if (prediction[0]==0):
        return '!!!!! The person is not diabetic !!!!!'
    else:
        return '!!!!! The person is diabetic !!!!!'
    
    
def main():
    
    
    #Giving a title for userInterface web page
    st.title('Diabetes Prediction Web App')
    
    
     #getting the input data from the user\
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('Skin Thickness Value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('Body Mass Index Value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
    Age=st.text_input('Age of the Person')
    
    # code for prediction
    diagnosis=''
    
    # creating a button for prediction
    if st.button('Diabetes test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
# when we run our program in anaconda navigator only this main function will run thas why we use this
# this file running directly

        
        
        
        
        
        
        
        
        
        
        