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



