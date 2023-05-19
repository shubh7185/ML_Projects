# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 

loaded_model=pickle.load(open('D:\projects\diabetes pridiction sysytem/trained_model.sav','rb'))
# in pytohn we use forward slash to save in directory
# rb = reading the file in binary 
# load model is used to load model

input_data=(3,126,88,41,235,39.3,0.704,27)
# we are changing the array into numpy array
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped )
print(prediction)
if (prediction[0]==0):
    print("!!!!! The person is not diabetic !!!!!")
else:
    print("!!!!! The person is diabetic !!!!!")