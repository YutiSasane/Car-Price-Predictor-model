
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AutoLux | Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# COLOR PALETTE
# White:       #FFFFFF
# Light Grey:  #E5E5E5
# Gold:        #FCA311
# Navy Blue:   #14213D
# Black:       #000000
# --------------------------------------------------

st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background-color: #14213D;
        color: #FFFFFF;
    }

    /* Main content width and spacing */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main heading */
    .main-title {
        color: #FFFFFF;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #FCA311;
        text-align: center;
        font-size: 1rem;
        letter-spacing: 3px;
        margin-bottom: 2rem;
    }

    /* Form card */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        padding: 2rem;
        border: 1px solid #E5E5E5;
        border-radius: 24px;
    }

    /* Form labels */
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] label p {
        color: #14213D !important;
        font-weight: 600 !important;
    }

    /* Input fields and dropdowns */
    div[data-testid="stForm"] input,
    div[data-testid="stForm"] div[data-baseweb="select"] > div {
        background-color: #E5E5E5 !important;
        color: #14213D !important;
        border-radius: 10px !important;
    }

    /* Gold prediction button */
    div[data-testid="stForm"] button[kind="primary"] {
        background-color: #FCA311 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        min-height: 48px;
        width: 100%;
    }

    div[data-testid="stForm"] button[kind="primary"]:hover {
        background-color: #E59600 !important;
        color: #000000 !important;
    }

    /* Prediction result */
    .result-card {
        background-color: #000000;
        border: 2px solid #FCA311;
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
    }

    .result-label {
        color: #E5E5E5;
        font-size: 1rem;
    }

    .result-price {
        color: #FCA311;
        font-size: 2rem;
        font-weight: 800;
    }

    .footer {
        color: #E5E5E5;
        text-align: center;
        font-size: 0.8rem;
        margin-top: 2rem;
    }

    /* General text */
    .stMarkdown, p {
        color: inherit;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# PROJECT FILES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_dataset():
    return pd.read_csv(BASE_DIR / "Cleaned_Car_data.csv")


@st.cache_resource
def load_model():
    with open(BASE_DIR / "LinearRegressionModel.pkl", "rb") as file:
        return pickle.load(file)


# Load dataset and trained model
try:
    car = load_dataset()
    price_model = load_model()

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error.filename}. "
        "Keep app.py, Cleaned_Car_data.csv, and "
        "LinearRegressionModel.pkl in the same directory."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the dataset or model: {error}")
    st.stop()


# Check that required dataset columns exist
required_columns = {
    "name",
    "company",
    "year",
    "kms_driven",
    "fuel_type",
}

missing_columns = required_columns - set(car.columns)

if missing_columns:
    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">AutoLux</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">CAR PRICE PREDICTOR</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="text-align:center; color:#E5E5E5;
    margin-bottom:2rem;">
        Discover the estimated value of your car.
        Enter its details below.
    </p>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DROPDOWN DATA
# --------------------------------------------------

companies = sorted(
    car["company"].dropna().unique().tolist(),
    key=str,
)

car_models = sorted(
    car["name"].dropna().unique().tolist(),
    key=str,
)

years = sorted(
    car["year"].dropna().unique().tolist(),
    reverse=True,
)

fuel_types = sorted(
    car["fuel_type"].dropna().unique().tolist(),
    key=str,
)

if not companies or not car_models or not years or not fuel_types:
    st.error("One or more dropdown lists are empty. Check your dataset.")
    st.stop()


# --------------------------------------------------
# KILOMETER OPTIONS
# --------------------------------------------------

km_options = [
    "Select common kilometers",
    "10000",
    "20000",
    "30000",
    "40000",
    "50000",
    "60000",
    "80000",
    "100000",
    "150000",
    "200000",
]


# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------

with st.form("car_prediction_form"):

    st.markdown(
        """
        <h3 style="color:#14213D; margin-top:0;">
            Enter Car Details
        </h3>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            "Car Company",
            options=companies,
            index=None,
            placeholder="Select company",
        )

        year = st.selectbox(
            "Manufacturing Year",
            options=years,
            index=None,
            placeholder="Select year",
        )

        fuel = st.selectbox(
            "Fuel Type",
            options=fuel_types,
            index=None,
            placeholder="Select fuel type",
        )

    with col2:
        model_name = st.selectbox(
            "Car Model",
            options=car_models,
            index=None,
            placeholder="Select car model",
        )

        common_km = st.selectbox(
            "Common Kilometers Driven",
            options=km_options,
            index=0,
        )

        manual_km = st.text_input(
            "Or Enter Kilometers Manually",
            placeholder="e.g. 45000",
            help="If both kilometer fields are filled, "
                 "the manual value takes priority.",
        )

    st.markdown(
        """
        <p style="color:#14213D; font-size:0.85rem;">
            Tip: Enter the total kilometers shown on your car's odometer.
        </p>
        """,
        unsafe_allow_html=True,
    )

    submitted = st.form_submit_button(
        "Predict Car Price",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    # Validate required selections
    if (
        company is None
        or model_name is None
        or year is None
        or fuel is None
    ):
        st.error("Please select the company, model, year, and fuel type.")

    else:
        # Manual input gets priority
        manual_km = manual_km.strip()

        if manual_km:
            km_value = manual_km

        elif common_km != "Select common kilometers":
            km_value = common_km

        else:
            km_value = ""

        if not km_value:
            st.warning("Please select or enter kilometers driven.")

        else:
            try:
                # Allow only whole-number kilometer values
                km = int(km_value)

                if km < 0:
                    st.error("Kilometers driven cannot be negative.")

                else:
                    # Keep the same feature order as the Flask model
                    input_data = pd.DataFrame(
                        [[model_name, company, int(year), km, fuel]],
                        columns=[
                            "name",
                            "company",
                            "year",
                            "kms_driven",
                            "fuel_type",
                        ],
                    )

                    # Predict price
                    prediction = float(
                        price_model.predict(input_data)[0]
                    )

                    if not pd.notna(prediction):
                        st.error("The model returned an invalid prediction.")

                    elif prediction < 0:
                        st.warning(
                            "The model estimated a negative price. "
                            "Please check the input and model training data."
                        )

                    else:
                        # Display estimated price
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-label">
                                    Estimated Car Price
                                </div>
                                <div class="result-price">
                                    ₹ {prediction:,.2f}
                                </div>
                                <p style="color:#E5E5E5; margin-bottom:0;">
                                    {company} {model_name}
                                    &nbsp;|&nbsp; {int(year)}
                                    &nbsp;|&nbsp; {km:,} km
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            except ValueError:
                st.error(
                    "Please enter a valid whole number for kilometers."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AUTO LUX &nbsp; • &nbsp;
        AI-POWERED CAR PRICE ESTIMATION
        <br>
        Predictions are estimates and not guaranteed selling prices.
    </div>
    """,
    unsafe_allow_html=True,
)
