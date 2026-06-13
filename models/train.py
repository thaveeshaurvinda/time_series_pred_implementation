import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

from data_preprocessing import (
    load_and_clean_data,
    handle_outliers_iqr,
    split_and_scale
)
from feature_engineering import (
    build_features,
    structure_3d_windows
)
from model import (
    get_baseline_model,
    get_lstm_model
)


def main():
    os.makedirs("reports", exist_ok = True)
    # Data pre processing
    print("oading and cleaning data...")
    raw_data = load_and_clean_data("data/energy_data_set.csv")

    # Remove outliers
    cleaned_data = handle_outliers_iqr(raw_data)

    # Split and scale data
    (train_data, test_data, feature_scaler, target_scaler, original_features) = split_and_scale(cleaned_data)

    # Feature engineering
    print("Creating new features...")
    # Create features for training data
    train_features = build_features(train_data)
    # Create features for testing data
    test_features = build_features(test_data)

    # Select columns for modelling
    modelling_features = [
        column
        for column in train_features.columns
        if column not in [
            "date",
            "Appliances",
            "hour",
            "target_lag_1",
            "target_lag_3",
            "target_roll_mean_6"
        ]
    ]

    # Prepare X and y datasets
    X_train = train_features[modelling_features].values
    y_train = train_features["Appliances"].values
    X_test = test_features[modelling_features].values
    y_test = test_features["Appliances"].values

    # Baseline model
    print("Training linear regression baseline...")
    # Create baseline model
    baseline_model = get_baseline_model()

    # Train model
    baseline_model.fit(X_train, y_train)

    # Make predictions
    baseline_predictions_scaled = (baseline_model.predict(X_test).reshape(-1, 1))

    # Convert scaled values back to original units
    baseline_predictions = (target_scaler.inverse_transform(baseline_predictions_scaled))

    # Prepare LSTM data
    # 24 records = 4 hours
    LOOKBACK_STEPS = 24

    # Create training sequences
    X_train_lstm, y_train_lstm = (structure_3d_windows(X_train, y_train, lookback_steps = LOOKBACK_STEPS))
    # Create testing sequences
    X_test_lstm, y_test_lstm = (structure_3d_windows(X_test, y_test, lookback_steps = LOOKBACK_STEPS))

    # Create LSTM model
    print("Creating and Compiling LSTM Network...")
    lstm_model = get_lstm_model(input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2]))

    # Early stopping
    early_stopping = EarlyStopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = True
    )

    # Train LSTM model
    print("Training Deep Learning Model...")
    lstm_model.fit(
        X_train_lstm,
        y_train_lstm,
        validation_split = 0.1,
        epochs = 30,
        batch_size = 32,
        callbacks = [early_stopping],
        verbose = 1
    )

    # Make predictions
    print("Generating Predictions...")
    # Predict on test data
    lstm_predictions_scaled = (lstm_model.predict(X_test_lstm))

    # Convert predictions back to original scale
    lstm_predictions = (target_scaler.inverse_transform(lstm_predictions_scaled))

    # Convert actual values back to original scale
    actual_values = (target_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)))

    # Align baseline predictions
    baseline_predictions_aligned = (
        baseline_predictions[
            LOOKBACK_STEPS:
        ]
    )

    # Calculate performance metrics
    # Linear regression
    baseline_mae = mean_absolute_error(
        actual_values,
        baseline_predictions_aligned
    )
    baseline_rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            baseline_predictions_aligned
        )
    )

    # LSTM
    lstm_mae = mean_absolute_error(
        actual_values,
        lstm_predictions
    )
    lstm_rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            lstm_predictions
        )
    )

    # Metric transformation (Classification report)
    print("Transforming continuous predictions into operational load classes...")
    # Define a clear operational boundary based on the home's median background usage (60 Wh)
    LOAD_THRESHOLD = 60.0

    # Convert continuous actual Watt hours into binary states:
    # 0 = Low Load State (Standby mode / Background appliances)
    # 1 = High Load State (Active human household appliance use)
    actual_classes = np.where(actual_values.flatten() <= LOAD_THRESHOLD, 0, 1)

    # Convert continuous Linear Regression predictions into classes
    baseline_classes = np.where(baseline_predictions_aligned.flatten() <= LOAD_THRESHOLD, 0, 1)

    # Convert continuous Deep Learning LSTM predictions into classes
    lstm_classes = np.where(lstm_predictions.flatten() <= LOAD_THRESHOLD, 0, 1)

    # Human readable labels for your technical evaluation printout
    class_labels = ["Low Load (<=60Wh)", "High Load (>60Wh)"]

    # Generate printouts for both architectures
    print("\n" + "=" * 65)
    print(f"{'LINEAR REGRESSION BASELINE CLASSIFICATION REPORT':^65}")
    print("=" * 65)
    print(classification_report(actual_classes, baseline_classes, target_names=class_labels))

    print("\n" + "=" * 65)
    print(f"{'DEEP LEARNING LSTM NETWORK CLASSIFICATION REPORT':^65}")
    print("=" * 65)
    print(classification_report(actual_classes, lstm_classes, target_names=class_labels))
    print("=" * 65)

    print("\n" + "=" * 55)
    print(f"{'Final performance evaluation table':^55}")
    print("=" * 55)
    print(
        f"Linear regression baseline = "
        f"MAE: {baseline_mae:.2f} Wh | "
        f"RMSE: {baseline_rmse:.2f} Wh"
    )
    print(
        f"LSTM Model = "
        f"MAE: {lstm_mae:.2f} Wh | "
        f"RMSE: {lstm_rmse:.2f} Wh"
    )
    print("=" * 55)

    # Actual VS predicted
    plt.figure(figsize = (14, 5))
    plt.plot(actual_values[:288], label = "Actual values", color = "black", alpha = 0.7, linewidth = 2)
    plt.plot(lstm_predictions[:288], label = "LSTM predictions", color = "dodgerblue", linestyle = "--", linewidth = 2)
    plt.title("Actual vs predicted appliance energy usage")
    plt.xlabel("Time steps")
    plt.ylabel("Energy usage (Wh)")
    plt.legend()
    plt.grid(True,alpha = 0.3)
    plt.savefig("reports/predicted_vs_actual.png", bbox_inches = "tight")

    # Residual plot
    plt.figure(figsize = (14, 4))
    residual_errors = (actual_values - lstm_predictions)
    plt.scatter(lstm_predictions, residual_errors, alpha = 0.15, color = "crimson")
    plt.axhline(y = 0, color = "black", linestyle = "-")
    plt.title("Residual error distribution")
    plt.xlabel("Predicted values (Wh)")
    plt.ylabel("Residual error (Wh)")
    plt.savefig("reports/residual_plot.png", bbox_inches = "tight")
    print("Graphs saved successfully inside the reports folder.")


if __name__ == "__main__":
    main()

