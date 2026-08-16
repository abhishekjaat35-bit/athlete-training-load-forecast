import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("=" * 80)
print("             ATHLETE TRAINING LOAD FORECAST")
print("=" * 80)


# ------------------------------------------
# Load Data
# ------------------------------------------

data = pd.read_csv("training_load_forecast_data.csv")

data["Date"] = pd.to_datetime(data["Date"])

data = data.sort_values(
    ["Athlete", "Date"]
).reset_index(drop=True)


# ------------------------------------------
# Data Validation
# ------------------------------------------

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

print(f"Rows           : {len(data)}")
print(f"Columns        : {len(data.columns)}")
print(
    f"Missing values : "
    f"{data.isnull().sum().sum()}"
)


# ------------------------------------------
# Create Session Number
# ------------------------------------------

data["Session_Number"] = (
    data.groupby("Athlete")
    .cumcount() + 1
)


# ------------------------------------------
# Create Lag Features
# ------------------------------------------

data["Previous_Load"] = (
    data.groupby("Athlete")["Training_Load"]
    .shift(1)
)

data["Load_2_Sessions_Ago"] = (
    data.groupby("Athlete")["Training_Load"]
    .shift(2)
)

data["Load_3_Sessions_Ago"] = (
    data.groupby("Athlete")["Training_Load"]
    .shift(3)
)


# ------------------------------------------
# Rolling Load Features
# ------------------------------------------

data["Rolling_Average_3"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform(
        lambda x:
        x.shift(1).rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)

data["Rolling_Average_5"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform(
        lambda x:
        x.shift(1).rolling(
            window=5,
            min_periods=5
        ).mean()
    )
)


# ------------------------------------------
# Prepare Modeling Dataset
# ------------------------------------------

model_data = data.dropna(
    subset=[
        "Previous_Load",
        "Load_2_Sessions_Ago",
        "Load_3_Sessions_Ago",
        "Rolling_Average_3",
        "Rolling_Average_5"
    ]
).copy()


features = [
    "Previous_Load",
    "Load_2_Sessions_Ago",
    "Load_3_Sessions_Ago",
    "Rolling_Average_3",
    "Rolling_Average_5"
]


# ------------------------------------------
# Train/Test Split
# ------------------------------------------

train_predictions = []
test_predictions = []


print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)


for athlete in model_data["Athlete"].unique():

    athlete_data = model_data[
        model_data["Athlete"] == athlete
    ].copy()

    athlete_data = athlete_data.sort_values(
        "Date"
    )

    split_index = int(
        len(athlete_data) * 0.75
    )

    train = athlete_data.iloc[
        :split_index
    ]

    test = athlete_data.iloc[
        split_index:
    ]

    if len(train) < 2 or len(test) < 1:
        continue

    X_train = train[features]
    y_train = train["Training_Load"]

    X_test = test[features]
    y_test = test["Training_Load"]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    for index, prediction in zip(
        test.index,
        predictions
    ):

        test_predictions.append(
            {
                "Index": index,
                "Athlete":
                    data.loc[index, "Athlete"],
                "Date":
                    data.loc[index, "Date"],
                "Actual_Load":
                    data.loc[index, "Training_Load"],
                "Predicted_Load":
                    prediction
            }
        )


# ------------------------------------------
# Prediction Results
# ------------------------------------------

results = pd.DataFrame(
    test_predictions
)


if len(results) > 0:

    results["Prediction_Error"] = (
        results["Actual_Load"]
        -
        results["Predicted_Load"]
    )

    results["Absolute_Error"] = (
        results["Prediction_Error"]
        .abs()
    )


    # --------------------------------------
    # Model Evaluation
    # --------------------------------------

    mae = mean_absolute_error(
        results["Actual_Load"],
        results["Predicted_Load"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            results["Actual_Load"],
            results["Predicted_Load"]
        )
    )


    print(
        f"Mean Absolute Error : "
        f"{mae:.2f} AU"
    )

    print(
        f"Root Mean Squared Error : "
        f"{rmse:.2f} AU"
    )


    # --------------------------------------
    # Display Predictions
    # --------------------------------------

    print("\n" + "=" * 80)
    print("FORECAST RESULTS")
    print("=" * 80)

    display_results = results.copy()

    display_results[
        "Predicted_Load"
    ] = display_results[
        "Predicted_Load"
    ].round(1)

    display_results[
        "Prediction_Error"
    ] = display_results[
        "Prediction_Error"
    ].round(1)

    print(
        display_results.to_string(
            index=False
        )
    )


else:

    print(
        "Not enough observations "
        "for model evaluation."
    )


# ------------------------------------------
# Train Final Models
# ------------------------------------------

future_predictions = []


for athlete in data["Athlete"].unique():

    athlete_data = data[
        data["Athlete"] == athlete
    ].copy()

    athlete_data = athlete_data.sort_values(
        "Date"
    )

    usable = athlete_data.dropna(
        subset=features
    )

    if len(usable) < 2:
        continue

    X = usable[features]
    y = usable["Training_Load"]

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    latest = usable.iloc[
        [-1]
    ]

    prediction = model.predict(
        latest[features]
    )[0]

    future_predictions.append(
        {
            "Athlete": athlete,
            "Last_Date":
                latest["Date"].iloc[0],
            "Last_Training_Load":
                latest["Training_Load"].iloc[0],
            "Predicted_Next_Load":
                prediction
        }
    )


# ------------------------------------------
# Future Forecast
# ------------------------------------------

future_forecast = pd.DataFrame(
    future_predictions
)


print("\n" + "=" * 80)
print("NEXT SESSION FORECAST")
print("=" * 80)


if len(future_forecast) > 0:

    future_forecast[
        "Predicted_Next_Load"
    ] = future_forecast[
        "Predicted_Next_Load"
    ].round(1)

    print(
        future_forecast.to_string(
            index=False
        )
    )

else:

    print(
        "Insufficient historical data "
        "for forecasting."
    )


# ------------------------------------------
# Visualization
# ------------------------------------------

plt.figure(figsize=(11, 6))

for athlete in data["Athlete"].unique():

    athlete_data = data[
        data["Athlete"] == athlete
    ]

    plt.plot(
        athlete_data["Date"],
        athlete_data["Training_Load"],
        marker="o",
        label=athlete
    )


plt.title(
    "Athlete Training Load Trend"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Training Load (AU)"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_load_forecast.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# Export Results
# ------------------------------------------

data.to_csv(
    "forecast_results.csv",
    index=False
)

if len(results) > 0:

    results.to_csv(
        "model_evaluation_results.csv",
        index=False
    )

if len(future_forecast) > 0:

    future_forecast.to_csv(
        "next_training_load_forecast.csv",
        index=False
    )


# ------------------------------------------
# Final Output
# ------------------------------------------

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("Generated files:")

print("1. forecast_results.csv")
print("2. training_load_forecast.png")

if len(results) > 0:
    print("3. model_evaluation_results.csv")

if len(future_forecast) > 0:
    print("4. next_training_load_forecast.csv")

print("\n" + "=" * 80)
print("MEASURE • MODEL • EVALUATE • FORECAST")
print("=" * 80)