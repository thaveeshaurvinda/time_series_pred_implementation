# Smart Multivariate Time-Series Appliance Load Forecasting Pipeline

An end-to-end Machine Learning and Deep Learning pipeline developed in Python to forecast household appliance energy consumption (Wh). The project compares the performance of a traditional Linear Regression model with a Long Short-Term Memory (LSTM) neural network for time-series forecasting. The pipeline includes data preprocessing, feature engineering, model training, evaluation, visualization, and a load-state classification system for smart-grid monitoring.

---

## Project Architecture

```text
├── data/
│   └── energy_data_set.csv        # Household energy consumption dataset
│
├── reports/
│   ├── outlier_boxplot.png        # Outlier analysis visualization
│   ├── predicted_vs_actual.png    # Forecast vs actual comparison plot
│   └── residual_plot.png          # Residual error analysis plot
│
├── data_preprocessing.py          # Data cleaning and scaling
├── feature_engineering.py         # Feature generation and sequence creation
├── model.py                       # Linear Regression and LSTM models
├── train.py                       # Main training and evaluation pipeline
└── README.md                      # Project documentation
```

---

## Key Features

### Data Preprocessing

* Converts date values into datetime format
* Sorts records chronologically
* Handles missing values using linear interpolation
* Detects and caps outliers using the Interquartile Range (IQR) method
* Applies Min-Max scaling for machine learning compatibility

### Feature Engineering

* Extracts hour information from timestamps
* Generates cyclical time features using sine and cosine transformations
* Creates weekend indicators
* Creates historical lag features:

  * `target_lag_1`
  * `target_lag_3`
* Generates rolling average features:

  * `target_roll_mean_6`
* Builds an indoor comfort index using temperature and humidity measurements

### Machine Learning Models

* Ordinary Least Squares (OLS) Linear Regression
* Long Short-Term Memory (LSTM) Recurrent Neural Network

### Time-Series Sequence Generation

* Converts 2D tabular data into 3D sequences
* Supports configurable lookback windows
* Produces data in the format required by LSTM models:

```text
(samples, time_steps, features)
```

### Smart Load-State Classification

Regression predictions are converted into operational load states using a threshold value.

Example:

```text
Energy Consumption < Threshold  → Normal Load
Energy Consumption ≥ Threshold  → High Load
```

This allows the forecasting system to be used as a smart-grid monitoring and alerting tool.

---

## Technical Highlights

### Cyclical Time Encoding

Time is transformed using sine and cosine functions to preserve the cyclical nature of hours.

```text
23:00 → Close to → 00:00
```

This prevents machine learning models from treating midnight as a large numerical jump.

### Historical Pattern Learning

Lag features and rolling averages help the models learn:

* Recent appliance usage behavior
* Consumption trends
* Short-term temporal dependencies

### Environmental Interaction Feature

The Indoor Comfort Index combines temperature and humidity:

```text
Indoor Comfort Index = Temperature × Humidity
```

This helps capture environmental effects on appliance energy consumption.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* TensorFlow / Keras
* Matplotlib
* Seaborn

---

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/appliance-load-forecasting.git
cd appliance-load-forecasting
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

### 3. Dataset Placement

Place the dataset inside the `data` directory:

```text
data/
└── energy_data_set.csv
```

---

## Running the Project

Execute the complete forecasting pipeline:

```bash
python train.py
```

This will:

1. Load and clean the dataset
2. Handle missing values
3. Detect and cap outliers
4. Generate engineered features
5. Create LSTM sequences
6. Train Linear Regression and LSTM models
7. Generate predictions
8. Evaluate model performance
9. Generate visual reports

---

## Model Evaluation Metrics

The models are evaluated using:

### Regression Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score (Coefficient of Determination)

### Classification Metrics

Load-state predictions are evaluated using:

* Precision
* Recall
* F1-Score
* Accuracy

---
