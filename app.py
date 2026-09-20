from flask import Flask, render_template, request
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
car = pd.read_csv(BASE_DIR / "Cleaned_Car_data.csv")

# Load trained model
with open(BASE_DIR / "LinearRegressionModel.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():

    # Dropdown data
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_types = sorted(car["fuel_type"].unique())

    # Default values
    prediction_text = ""

    selected_company = ""
    selected_model = ""
    selected_year = ""
    selected_fuel = ""
    selected_km = ""

    # When user clicks Predict Price
    if request.method == "POST":

        # Get values from form
        company = request.form.get("company")
        model_name = request.form.get("car_model")
        year = int(request.form.get("year"))
        fuel = request.form.get("fuel_type")

        # Get both kilometer values
        manual_km = request.form.get("kilo_driven", "").strip()
        common_km = request.form.get("km_options", "").strip()

        # Choose kilometer value
        # Manual input gets priority if both are filled
        if manual_km:
            km_value = manual_km

        elif common_km:
            km_value = common_km

        else:
            km_value = ""

        # Remember user's selections
        selected_company = company
        selected_model = model_name
        selected_year = str(year)
        selected_fuel = fuel
        selected_km = km_value

        # Check kilometer value
        if not km_value:

            prediction_text = "Please select or enter kilometers driven."

        else:

            try:
                km = int(km_value)

                # Create input dataframe
                input_data = pd.DataFrame(
                    [[model_name, company, year, km, fuel]],
                    columns=[
                        "name",
                        "company",
                        "year",
                        "kms_driven",
                        "fuel_type"
                    ]
                )

                # Predict price
                prediction = model.predict(input_data)[0]

                # Display prediction
                prediction_text = f"Estimated Price: ₹ {round(prediction, 2)}"

            except ValueError:

                prediction_text = "Please enter a valid number for kilometers."

    # Send data to HTML
    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
        prediction_text=prediction_text,
        selected_company=selected_company,
        selected_model=selected_model,
        selected_year=selected_year,
        selected_fuel=selected_fuel,
        selected_km=selected_km
    )


if __name__ == "__main__":
    app.run(debug=True)