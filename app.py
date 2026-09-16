import streamlit as st
import pickle
import pandas as pd

from src.preprocess import load_data, preprocess_data, split_data


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Coal Production Prediction",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 15px;
}

.prediction-box {
    padding: 25px;
    border-radius: 12px;
    background-color: #f5f7fa;
    border: 1px solid #ddd;
    text-align: center;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
}

.metric-label {
    font-size: 15px;
    color: #666;
}

.footer {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_training_columns():
    df = load_data()
    df = preprocess_data(df)
    X, y = split_data(df)
    return X.columns.tolist()


model = load_model()
training_columns = load_training_columns()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="title">⛏️ Coal Production Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based prediction of coal production using '
    'geological, ownership and geographical features.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("About the Model")

    st.write(
        "This application uses a machine learning model trained "
        "on historical coal production and mine-related data."
    )

    st.divider()

    st.subheader("Input Features")

    st.write("• State / UT")
    st.write("• Coal / Lignite")
    st.write("• Ownership")
    st.write("• Latitude")
    st.write("• Longitude")

    st.divider()

    st.caption(
        "Prediction results are estimates based on historical "
        "data patterns and model training."
    )


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Mine Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    state = st.text_input(
        "State / UT Name",
        placeholder="e.g. Maharashtra"
    )

    coal_type = st.selectbox(
        "Coal / Lignite",
        ["Coal", "Lignite"]
    )

    ownership = st.selectbox(
        "Ownership",
        ["Govt Owned", "Private"]
    )


with col2:

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=20.0,
        step=0.0001,
        format="%.4f"
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=78.0,
        step=0.0001,
        format="%.4f"
    )


st.divider()


# ---------------------------------------------------------
# INPUT PREPARATION
# ---------------------------------------------------------

input_dict = {
    "State/UT Name": state.strip(),
    "Coal/Lignite": coal_type,
    "Govt Owned/Private": ownership,
    "Latitude ": latitude,
    "Longitude ": longitude
}

input_df = pd.DataFrame([input_dict])


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

predict_button = st.button(
    "🔍 Predict Coal Production",
    use_container_width=True,
    type="primary"
)


if predict_button:

    # Validate state
    if not state.strip():

        st.warning("Please enter a State / UT Name.")

    else:

        try:

            # Convert categorical variables
            input_encoded = pd.get_dummies(input_df)

            # Match training features
            input_encoded = input_encoded.reindex(
                columns=training_columns,
                fill_value=0
            )

            # Prediction
            prediction = model.predict(input_encoded)[0]

            # -------------------------------------------------
            # IMPORTANT:
            # Replace this conversion with inverse_transform()
            # if a target scaler was used during training.
            # -------------------------------------------------

            real_prediction = prediction * 45

            # Production category
            if real_prediction < 10:
                level = "Low Production"
            elif real_prediction < 25:
                level = "Medium Production"
            else:
                level = "High Production"


            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            st.success("Prediction generated successfully.")

            st.markdown(
                '<div class="section-title">Prediction Result</div>',
                unsafe_allow_html=True
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    label="Estimated Coal Production",
                    value=f"{real_prediction:.2f} MT"
                )

            with result_col2:

                st.metric(
                    label="Production Level",
                    value=level
                )


            # -------------------------------------------------
            # INPUT SUMMARY
            # -------------------------------------------------

            st.divider()

            st.subheader("Input Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.write("**State / UT**")
                st.write(state)

            with summary_col2:
                st.write("**Coal Type**")
                st.write(coal_type)

            with summary_col3:
                st.write("**Ownership**")
                st.write(ownership)


            summary_col4, summary_col5 = st.columns(2)

            with summary_col4:
                st.write("**Latitude**")
                st.write(f"{latitude:.4f}")

            with summary_col5:
                st.write("**Longitude**")
                st.write(f"{longitude:.4f}")


            # -------------------------------------------------
            # INTERPRETATION
            # -------------------------------------------------

            st.divider()

            st.subheader("Prediction Interpretation")

            if level == "Low Production":

                st.info(
                    "The model estimates relatively low coal production "
                    "for the provided mine characteristics."
                )

            elif level == "Medium Production":

                st.info(
                    "The model estimates a moderate level of coal "
                    "production based on the provided characteristics."
                )

            else:

                st.info(
                    "The model estimates relatively high coal production "
                    "for the provided mine characteristics."
                )


            st.caption(
                "Note: This prediction is generated from historical "
                "training data and should be interpreted as an estimate, "
                "not as an exact production value."
            )


        except Exception as e:

            st.error(
                "Unable to generate the prediction. "
                "Please verify the input values and model configuration."
            )

            st.exception(e)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer">'
    'Coal Production Prediction System • Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)