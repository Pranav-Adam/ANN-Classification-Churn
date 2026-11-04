
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -----------------------------------------
# 1. Load trained model
# -----------------------------------------

model = tf.keras.models.load_model('model.h5')


# -----------------------------------------
# 2. Load encoders and scaler
# -----------------------------------------

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# -----------------------------------------
# 3. Streamlit application
# -----------------------------------------

st.title('Customer Churn Prediction')

st.write(
    'Enter customer details to predict whether the customer '
    'is likely to leave the bank.'
)


# -----------------------------------------
# 4. Customer input fields
# -----------------------------------------

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider(
    'Age',
    min_value=18,
    max_value=92,
    value=18
)

balance = st.number_input(
    'Balance',
    min_value=0.0,
    value=0.0
)

credit_score = st.number_input(
    'Credit Score',
    min_value=0,
    max_value=1000,
    value=0
)

estimated_salary = st.number_input(
    'Estimated Salary',
    min_value=0.0,
    value=0.0
)

tenure = st.slider(
    'Tenure',
    min_value=0,
    max_value=10,
    value=0
)

num_of_products = st.slider(
    'Number of Products',
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)


# -----------------------------------------
# 5. Initial probability
# -----------------------------------------

if credit_score == 0:

    st.write('### Churn Probability: 0.00')
    st.write('### Churn Percentage: 0.00%')

    st.info('Enter a valid Credit Score to start prediction.')


# -----------------------------------------
# 6. Validate customer input
# -----------------------------------------

elif credit_score < 300 or credit_score > 900:

    st.error('Credit Score must be between 300 and 900.')


else:

    # -------------------------------------
    # 7. Prepare input DataFrame
    # -------------------------------------

    input_data = pd.DataFrame({

        'CreditScore': [credit_score],

        'Gender': [
            label_encoder_gender.transform([gender])[0]
        ],

        'Age': [age],

        'Tenure': [tenure],

        'Balance': [balance],

        'NumOfProducts': [num_of_products],

        'HasCrCard': [has_cr_card],

        'IsActiveMember': [is_active_member],

        'EstimatedSalary': [estimated_salary]

    })


    # -------------------------------------
    # 8. One-hot encode Geography
    # -------------------------------------

    geography_input = pd.DataFrame({
        'Geography': [geography]
    })

    geo_encoded = onehot_encoder_geo.transform(
        geography_input
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(
            ['Geography']
        )
    )


    # -------------------------------------
    # 9. Combine customer and geography data
    # -------------------------------------

    input_data = pd.concat(
        [input_data, geo_encoded_df],
        axis=1
    )


    # -------------------------------------
    # 10. Match scaler feature order
    # -------------------------------------

    if hasattr(scaler, 'feature_names_in_'):

        expected_features = list(
            scaler.feature_names_in_
        )

        missing_features = [
            col for col in expected_features
            if col not in input_data.columns
        ]

        if missing_features:

            st.error(
                f'Missing features: {missing_features}'
            )

            st.stop()

        input_data = input_data[expected_features]


    # -------------------------------------
    # 11. Scale input data
    # -------------------------------------

    input_data_scaled = scaler.transform(input_data)


    # -------------------------------------
    # 12. Reshape input data
    # -------------------------------------

    # Convert (1, 12) into (1, 12, 1)

    input_data_reshaped = input_data_scaled.reshape(
        input_data_scaled.shape[0],
        input_data_scaled.shape[1],
        1
    )


    # -------------------------------------
    # 13. Check model input shape
    # -------------------------------------

    if input_data_reshaped.shape[1:] != tuple(
        model.input_shape[1:]
    ):

        st.error(
            f'Model expects {model.input_shape}, '
            f'but received {input_data_reshaped.shape}.'
        )

        st.stop()


    # -------------------------------------
    # 14. Predict customer churn
    # -------------------------------------

    prediction = model.predict(
        input_data_reshaped,
        verbose=0
    )

    prediction_proba = float(
        np.asarray(prediction).reshape(-1)[0]
    )


    # -------------------------------------
    # 15. Display prediction
    # -------------------------------------

    st.write(
        f'### Churn Probability: {prediction_proba:.2f}'
    )

    st.write(
        f'### Churn Percentage: {prediction_proba * 100:.2f}%'
    )


    # -------------------------------------
    # 16. Display churn result
    # -------------------------------------

    if prediction_proba > 0.5:

        st.error(
            'The customer is likely to churn.'
        )

    else:

        st.success(
            'The customer is not likely to churn.'
        )
