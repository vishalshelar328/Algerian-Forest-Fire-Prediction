import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import the Ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('Models/ridge1.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

# Set up the Streamlit app
def main():
    st.title('Welcome to Homepage')

    # Display input fields
    temperature = st.text_input('Temperature', value='')
    rh = st.text_input('RH', value='')
    ws = st.text_input('Ws', value='')
    rain = st.text_input('Rain', value='')
    ffmc = st.text_input('FFMC', value='')
    dmc = st.text_input('DMC', value='')
    isi = st.text_input('ISI', value='')
    classes = st.text_input('Classes', value='')
    region = st.text_input('Region', value='')

    if st.button('Predict'):
        new_data_scaled = standard_scaler.transform(
            np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]]))
        result = ridge_model.predict(new_data_scaled)

        # Display the prediction result
        st.header('FWI Prediction')
        st.write(f'The FWI prediction is {result[0]}')

if __name__ == '__main__':
    main()
