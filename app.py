
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.preprocess import load_data, preprocess_data, split_data


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

st.set_page_config(
    page_title="Coal Analytics | Production Predictor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #172333;
    }

    [data-testid="stSidebar"] * {
        color: #f1f5f9;
    }

    .hero {
        background: linear-gradient(120deg, #172333, #263f55);
        padding: 30px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        color: white;
    }

    .hero h1 {
        font-size: 32px;
        font-weight: 750;
        margin: 0 0 10px 0;
        color: white;
    }

    .hero p {
        color: #d4e0ec;
        font-size: 15px;
        margin: 0;
        line-height: 1.7;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 700;
        color: #172333;
        margin: 12px 0 5px 0;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        min-height: 125px;
    }

    .result-label {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 10px;
    }

    .result-value {
        font-size: 27px;
        font-weight: 750;
        color: #172333;
        overflow-wrap: anywhere;
    }

    .result-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 8px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 20px 0 8px 0;
    }

    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px;
    }

    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] button {
        background-color: #176b52;
        border: 1px solid #176b52;
        color: white;
        font-weight: 650;
        min-height: 46px;
        border-radius: 9px;
    }

    div.stButton > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #125640;
        border-color: #125640;
        color: white;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL AND DATA LOADING
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_training_columns():
    df = load_data()
    df = preprocess_data(df)
    X, _ = split_data(df)

    return X.columns.tolist()


try:
    model = load_model()
    training_columns = load_training_columns()

except Exception:
    st.error(
        "The application could not load its model or training data. "
        "Check the model path and preprocessing configuration."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## ⛏️ COAL ANALYTICS")
    st.caption("MACHINE LEARNING PLATFORM")

    st.divider()

    st.markdown("### About the Project")
    st.write(
        "A machine-learning application designed to estimate "
        "coal production from mine ownership, resource type, "
        "state and geographical coordinates."
    )

    st.markdown("### Model Inputs")

    st.markdown(
        """
        - State / Union Territory
        - Coal or lignite classification
        - Ownership category
        - Latitude and longitude
        """
    )

    st.divider()

    st.markdown("### Important Information")
    st.caption(
        "Predictions depend on the training data and model "
        "quality. They are estimates, not verified production "
        "records or official forecasts."
    )

    st.markdown("---")
    st.caption("Coal Analytics | Academic ML Project")


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>Coal Production Predictor</h1>
        <p>
            Estimate coal production using machine learning
            and mine-related geographical and ownership features.
            Enter the mine information below to generate a prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INPUT FORM
# =========================================================

st.markdown(
    '<div class="section-heading">Mine Information</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Provide the available mine characteristics to estimate production.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        state = st.text_input(
            "State / Union Territory",
            placeholder="e.g. Maharashtra",
            help="Enter a state name consistent with the training dataset.",
        )

        coal_type = st.selectbox(
            "Resource Classification",
            ["Coal", "Lignite"],
        )

        ownership = st.selectbox(
            "Ownership Type",
            ["Govt Owned", "Private"],
        )

    with col2:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=20.0,
            step=0.0001,
            format="%.4f",
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.0,
            step=0.0001,
            format="%.4f",
        )

        st.caption(
            "Coordinates should correspond to the mine's location."
        )

    submitted = st.form_submit_button(
        "Generate Production Estimate",
        use_container_width=True,
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:
    if not state.strip():
        st.warning(
            "Please enter a State / Union Territory before continuing."
        )

    else:
        try:
            # Keep the column names consistent with the training data.
            input_data = pd.DataFrame(
                [
                    {
                        "State/UT Name": state.strip(),
                        "Coal/Lignite": coal_type,
                        "Govt Owned/Private": ownership,
                        "Latitude ": latitude,
                        "Longitude ": longitude,
                    }
                ]
            )

            # One-hot encode categorical features.
            input_encoded = pd.get_dummies(input_data)

            # Match the exact training feature structure.
            input_encoded = input_encoded.reindex(
                columns=training_columns,
                fill_value=0,
            )

            # Predict using the loaded model.
            prediction = float(model.predict(input_encoded)[0])

            if not pd.notna(prediction):
                raise ValueError("The model returned an invalid prediction.")

            # IMPORTANT:
            # Do not multiply by 45 unless this is the documented
            # inverse transformation used during model training.
            production = prediction

            # These thresholds are examples. Adjust them to match
            # the actual target unit and project requirements.
            if production < 10:
                production_level = "Low"
                interpretation = (
                    "The predicted value falls within the low "
                    "production band configured for this application."
                )
            elif production < 25:
                production_level = "Medium"
                interpretation = (
                    "The predicted value falls within the medium "
                    "production band configured for this application."
                )
            else:
                production_level = "High"
                interpretation = (
                    "The predicted value falls within the high "
                    "production band configured for this application."
                )

            st.divider()

            st.markdown(
                '<div class="section-heading">Prediction Results</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-description">'
                'Results generated from the supplied mine information.'
                '</div>',
                unsafe_allow_html=True,
            )

            # -------------------------------------------------
            # RESULT CARDS
            # -------------------------------------------------

            result_col1, result_col2 = st.columns(2, gap="large")

            with result_col1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            PREDICTED PRODUCTION
                        </div>
                        <div class="result-value">
                            {production:,.2f}
                        </div>
                        <div class="result-note">
                            Model output in the target's training unit
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_col2:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            PRODUCTION CATEGORY
                        </div>
                        <div class="result-value">
                            {production_level}
                        </div>
                        <div class="result-note">
                            Based on the configured thresholds
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # -------------------------------------------------
            # INPUT SUMMARY
            # -------------------------------------------------

            st.markdown("### Input Summary")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("State / UT", state.strip())

            with summary_col2:
                st.metric("Resource", coal_type)

            with summary_col3:
                st.metric("Ownership", ownership)

            location_col1, location_col2 = st.columns(2)

            with location_col1:
                st.metric("Latitude", f"{latitude:.4f}")

            with location_col2:
                st.metric("Longitude", f"{longitude:.4f}")

            # -------------------------------------------------
            # INTERPRETATION
            # -------------------------------------------------

            st.markdown("### Interpretation")
            st.info(interpretation)

            st.caption(
                "This result is a model-generated estimate. "
                "Its reliability depends on the quality and coverage "
                "of the training dataset. It should not be treated "
                "as an official production figure."
            )

            # -------------------------------------------------
            # EXPORT RESULT
            # -------------------------------------------------

            report = pd.DataFrame(
                [
                    {
                        "State/UT Name": state.strip(),
                        "Coal/Lignite": coal_type,
                        "Ownership": ownership,
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "Predicted Production": production,
                        "Production Category": production_level,
                    }
                ]
            )

            st.download_button(
                label="Download Prediction Report (CSV)",
                data=report.to_csv(index=False).encode("utf-8"),
                file_name="coal_production_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception:
            st.error(
                "Prediction failed. Verify the input features, "
                "model file, preprocessing steps and target transformation."
            )
            st.caption(
                "For debugging, check the application logs and terminal "
                "for the underlying exception."
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        COAL ANALYTICS &nbsp;•&nbsp;
        Machine Learning Based Production Estimation
    </div>
    """,
    unsafe_allow_html=True,
)
