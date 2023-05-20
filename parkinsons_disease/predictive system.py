

import numpy as np
import pickle

loaded_model=pickle.load(open('D:\projects\parkinsons_disease/trained_model.sav','rb'))


input_data=(119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)
input_data_as_numpyarray=np.asarray(input_data)
input_data_reshape=input_data_as_numpyarray.reshape(1,-1)
# std_data=sc.transform(input_data_reshape)
prediction=loaded_model.predict(input_data_reshape)
print(prediction)
if (prediction[0]==0):
  print('person does not have a parkinsons disease')
else:
  print('person have a parkinsons disease')