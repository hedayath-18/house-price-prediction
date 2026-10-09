# House Price Prediction

A machine learning project that predicts median house values using the California Housing dataset. This project compares a mean-prediction baseline with a Random Forest regression model.

## Project Overview

The goal is to build and evaluate a regression model using a reproducible train/test split.

## Tech Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Dataset

The project uses the California Housing dataset available through scikit-learn.

The target represents median house values in units of $100,000.

## Methodology

1. Load the dataset.
2. Split the data into training and test sets (80/20).
3. Train a mean-prediction baseline.
4. Train a Random Forest regression model.
5. Evaluate both models using MAE, RMSE, and R².
6. Save the trained model using Joblib.

## Results

Results from the reported test run:

| Metric | Mean Baseline | Random Forest |
|---|---:|---:|
| MAE | 0.9061 | 0.3306 |
| RMSE | 1.1449 | 0.5084 |
| R² | -0.0002 | 0.8028 |

The Random Forest performed better than the baseline across all three reported metrics on this test split.

### Interpreting the Results

- **MAE:** Mean absolute prediction error.
- **RMSE:** Root mean squared prediction error, which penalises larger errors more heavily.
- **R²:** Measures how well the model explains variation in the target relative to a mean-prediction baseline.

## Project Structure

```text
house-price-prediction/
├── src/
│   └── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model is saved to `models/house_price_model.joblib` when the script runs. Generated model artifacts are not included in the repository unless explicitly committed.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training script from the project root:

```bash
python src/train.py
```

The first run requires an internet connection to download the dataset.

## Limitations

- Results are based on one 80/20 random train/test split.
- Further validation, including cross-validation and error analysis, is needed.
- Dataset performance does not guarantee accurate predictions for current housing markets.
- The model is intended for learning and experimentation, not real-estate valuation decisions.

## Future Improvements

- Add unit tests.
- Perform cross-validation.
- Analyse feature importance and prediction errors.
- Add a prediction interface.
- Set up automated tests with GitHub Actions.

## Author

Hedayath Pinjari

GitHub: [@hedayath-18](https://github.com/hedayath-18)
