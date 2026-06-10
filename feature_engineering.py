import numpy as np


def build_features(data, target_column="Appliances"):
    """
    Creates additional features that may help the machine learning model learn patterns better.
    Features created:
    1. Cyclical hour features (sin and cos)
    2. Weekend indicator
    3. Historical lag values
    4. Rolling average
    5. Indoor comfort index
    """
    feature_data = data.copy()

    # Extract hour from date
    # Get hour from datetime column
    feature_data["hour"] = feature_data["date"].dt.hour

    # Convert hour into sine values
    feature_data["hour_sin"] = np.sin(
        2 * np.pi * feature_data["hour"] / 24.0
    )
    # Convert hour into cosine values
    feature_data["hour_cos"] = np.cos(
        2 * np.pi * feature_data["hour"] / 24.0
    )

    # Create weekend indicator
    # Day numbers:
    # Monday = 0
    # Tuesday = 1
    # Wednesday = 2
    # Thursday = 3
    # Friday = 4
    # Saturday = 5
    # Sunday = 6

    # Morning vs night
    # Weekday vs weekend
    # Previous energy usage trends
    # Environmental conditions

    feature_data["is_weekend"] = (
        feature_data["date"]
        .dt.dayofweek
        .apply(lambda day: 1.0 if day >= 5 else 0.0)
    )

    # Create historical lag features
    # Value from 1 step earlier (10 minutes ago)
    feature_data["target_lag_1"] = (
        feature_data[target_column].shift(1)
    )

    # Value from 3 steps earlier (30 minutes ago)
    feature_data["target_lag_3"] = (
        feature_data[target_column].shift(3)
    )

    # Create rolling mean feature
    # Rolling mean over previous 6 records
    # Uses shift(1) so current value is not included
    feature_data["target_roll_mean_6"] = (
        feature_data[target_column]
        .shift(1) # Use previous value as the first value in the rolling window (10 mins)
        .rolling(window = 6) # Look back 6 previous values (6x10 = 60 minutes)
        .mean()
    )

    # Create environmental interaction feature
    # Check if both columns exist
    if "T1" in feature_data.columns and "RH_1" in feature_data.columns:
        # Temperature x Humidity
        feature_data["indoor_comfort_index"] = (
            feature_data["T1"] *
            feature_data["RH_1"]
        )

    # Remove missing values
    # Lag and rolling operations create NaN values in the first few rows
    feature_data = feature_data.dropna()

    # Reset row numbers after removing missing values
    feature_data = feature_data.reset_index(drop = True)
    return feature_data

