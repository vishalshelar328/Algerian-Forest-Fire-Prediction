import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import the Ridge regressor model and standard scaler pickle
ridge_model = pickle.load(open('Models/ridge1.pkl', 'rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl', 'rb'))

def validate_input(value):
    if value not in ['0', '1']:
        raise ValueError('Values should be either 0 or 1.')

def main():
    st.markdown(
        """
        <style>
        body {
            background: url('path_to_your_image.jpg') no-repeat center center fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Fire Weather Index(FWI) Prediction')

    # Display input fields
    temperature = st.text_input('Temperature', value='')
    rh = st.text_input('RH', value='')
    ws = st.text_input('Ws', value='')
    rain = st.text_input('Rain', value='')
    ffmc = st.text_input('FFMC', value='')
    dmc = st.text_input('DMC', value='')
    isi = st.text_input('ISI', value='')

    region = st.text_input('Region', value='')
    try:
        validate_input(region)
    except ValueError as e:
        st.error(str(e))
        return  # Prevent prediction

    classes = st.text_input('Classes', value='')
    try:
        validate_input(classes)
    except ValueError as e:
        st.error(str(e))
        return  # Prevent prediction

    if st.button('Predict'):
        new_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]], dtype=float)
        new_data_scaled = standard_scaler.transform(new_data)
        result = ridge_model.predict(new_data_scaled)

        # Display the prediction result
        st.header('FWI Prediction')
        st.write(f'The FWI prediction is {result[0]}')

if __name__ == '__main__':
    main()
