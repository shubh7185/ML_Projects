import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:\projects\parkinsons_disease/trained_model.sav', 'rb'))

def parkinsons_prediction(input_data):
    input_data_as_numpyarray = np.asarray(input_data)
    input_data_reshape = input_data_as_numpyarray.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)
    if prediction[0] == 0:
        return "The person does not have Parkinson's disease."
    else:
        return "The person has Parkinson's disease."

def main():
    st.title("Parkinson's Disease Prediction Web App")
    
    fo_hz = st.text_input('Average vocal fundamental frequency')
    fhi_hz = st.text_input('Maximum vocal fundamental frequency')
    flo_hz = st.text_input('Minimum vocal fundamental frequency')
    jitter = st.text_input('Several measures of variation in fundamental frequency %')
    jitter_abs = st.text_input('Several measures of variation in fundamental frequency (ABS)')
    rap = st.text_input('Several measures of variation in fundamental frequency (RAP)')
    ppq = st.text_input('Several measures of variation in fundamental frequency (PPQ)')
    ddp = st.text_input('Several measures of variation in fundamental frequency (DDP)')
    shimmer = st.text_input('Several measures of variation in amplitude 1')
    shimmer_db = st.text_input('Several measures of variation in amplitude (DB)')
    shimmer_apq3 = st.text_input('Several measures of variation in amplitude (APQ3)')
    shimmer_apq5 = st.text_input('Several measures of variation in amplitude (APQ5)')
    apq = st.text_input('Several measures of variation in amplitude (APQ)')
    shimmer_dda = st.text_input('Several measures of variation in amplitude (DDA)')
    nhr = st.text_input('Two measures of the ratio of noise to tonal components in the voice 1')
    hnr = st.text_input('Two measures of the ratio of noise to tonal components in the voice 2')
    rpde = st.text_input('Two nonlinear dynamical complexity measures1')
    dfa = st.text_input('Signal fractal scaling exponent')
    spread1 = st.text_input('Three nonlinear measures of fundamental frequency variation1')
    spread2 = st.text_input('Three nonlinear measures of fundamental frequency variation2')
    d2 = st.text_input('Two nonlinear dynamical complexity measures2')
    ppe = st.text_input('Three nonlinear measures of fundamental frequency variation')

    diagnosis = ''

    if st.button('Parkinsons Test Result'):
        diagnosis = parkinsons_prediction([fo_hz, fhi_hz, flo_hz, jitter, jitter_abs, rap, ppq, ddp, shimmer,
                                           shimmer_db, shimmer_apq3, shimmer_apq5, apq, shimmer_dda, nhr, hnr,
                                           rpde, dfa, spread1, spread2, d2, ppe])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

