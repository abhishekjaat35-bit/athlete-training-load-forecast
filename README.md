# Athlete Training Load Forecast

A Python sports analytics project that uses historical athlete training-load data and linear regression to estimate future training load.

## Objective

The project demonstrates a basic predictive analytics workflow for athlete training-load data.

It:

- Loads longitudinal training data
- Validates the dataset
- Creates lagged training-load variables
- Creates rolling averages
- Trains a regression model
- Evaluates prediction error
- Forecasts the next training-load value
- Creates visualizations
- Exports analysis results

## Data Flow

```text
Historical Training Load
          ↓
Data Validation
          ↓
Lag Features
          ↓
Rolling Averages
          ↓
Linear Regression
          ↓
Model Evaluation
          ↓
Next-Session Forecast
```

## Dataset

The dataset contains:

| Variable | Description |
|---|---|
| Athlete | Athlete identifier |
| Date | Training date |
| Training_Load | Session training load in arbitrary units |

## Features

The model uses:

- Previous training load
- Training load two sessions ago
- Training load three sessions ago
- Three-session rolling average
- Five-session rolling average

## Model

The project uses linear regression.

The model learns relationships between previous training-load observations and the next observed training load.

Conceptually:

```text
Historical Load
      ↓
Features
      ↓
Linear Regression
      ↓
Predicted Load
```

## Model Evaluation

Two error metrics are calculated.

### Mean Absolute Error

```text
MAE = average absolute prediction error
```

MAE is expressed in the same units as training load.

### Root Mean Squared Error

```text
RMSE = square root of mean squared prediction error
```

RMSE gives greater influence to larger prediction errors.

## Train/Test Strategy

The dataset is ordered chronologically.

Earlier observations are used for training and later observations are used for testing.

This is preferable to randomly shuffling longitudinal training data because future observations should not be used to predict the past.

## Generated Files

```text
forecast_results.csv
model_evaluation_results.csv
next_training_load_forecast.csv
training_load_forecast.png
```

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Linear Regression
- Time-series feature engineering

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Running the Project

Place the Python script and CSV file in the same directory.

Run:

```bash
python athlete_training_load_forecast.py
```

## Sports Science Application

Predictive training-load analysis can potentially support:

- Training monitoring
- Load planning
- Athlete profiling
- Periodization analysis
- Performance analytics
- Decision-support systems

However, a statistical forecast should not automatically be interpreted as the appropriate training prescription.

## Limitations

This project uses synthetic data.

The dataset is small and deliberately structured for learning.

Linear regression assumes a relatively simple relationship between the predictors and outcome.

Real athlete training-load data are affected by many variables, including:

- Athlete readiness
- Sleep
- Wellness
- Recovery
- Competition schedule
- Travel
- Injury status
- Training phase
- Coaching decisions
- Performance goals

A forecast therefore represents a statistical estimate rather than a coaching recommendation.

## Future Development

Potential improvements include:

- Larger longitudinal datasets
- Athlete-specific models
- Random forest regression
- Gradient boosting
- XGBoost
- Time-series models
- Exponential smoothing
- Bayesian forecasting
- Readiness variables
- GPS load
- Heart-rate load
- Wellness data
- Performance testing
- Automated model selection
- Model monitoring
- Interactive dashboards

## Skills Demonstrated

```text
Python
   ↓
Pandas
   ↓
Feature Engineering
   ↓
Time-Series Data
   ↓
Machine Learning
   ↓
Regression
   ↓
Model Evaluation
   ↓
Forecasting
```

## Author

**Abhishek Tomar**

Strength & Conditioning | Sports Performance | Sports Analytics | Python

## License

MIT License