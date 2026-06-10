import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Load and clean the data
def load_and_clean_data(file_path):
    """
    1. Reads the CSV file
    2. Converts date column into datetime format
    3. Sorts records by date
    4. Handles missing values using interpolation
    """
    data = pd.read_csv(file_path)

    # Convert date column into proper datetime format
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    # Reset index after sorting
    data = data.reset_index(drop = True)

    # Check if there are missing values
    total_missing_values = data.isnull().sum().sum()

    if total_missing_values > 0:
        # Fill missing values smoothly via linear interpolation, then backfill/forwardfill boundary edges
        data = data.interpolate(method = "linear").bfill().ffill()
    return data


# Handle outliers using IQR
def handle_outliers_iqr(data, target_column = "Appliances"):
    """
    Finds outliers using the IQR method.
    Replace extreme values with the upper or lower limit.
    """
    # Create a copy so original data is unchanged
    cleaned_data = data.copy()

    Q1 = cleaned_data[target_column].quantile(0.25)
    Q3 = cleaned_data[target_column].quantile(0.75)
    # Calculate Interquartile Range
    IQR = Q3 - Q1

    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    # Replace values above upper limit
    cleaned_data[target_column] = np.where(
        cleaned_data[target_column] > upper_limit,
        upper_limit,
        cleaned_data[target_column]
    )

    # Replace values below lower limit
    cleaned_data[target_column] = np.where(
        cleaned_data[target_column] < lower_limit,
        lower_limit,
        cleaned_data[target_column]
    )
    return cleaned_data


# Split data and scale
def split_and_scale(data, target_column = "Appliances", train_ratio = 0.8):
    """
    1. Splits data into training and testing sets
    2. Keeps chronological order
    3. Applies Min-Max scaling
    """
    # Columns that should not be used as features (inputs)
    columns_to_remove = ["date", target_column, "WeekStatus"]

    feature_columns = []
    for column in data.columns:
        if column not in columns_to_remove:
            feature_columns.append(column)

    # Calculate split position
    split_index = int(len(data) * train_ratio)

    # Training data = first 80%
    train_data = data.iloc[:split_index].copy()
    # Testing data = remaining 20%
    test_data = data.iloc[split_index:].copy()

    # Create scaler for features (X)
    feature_scaler = MinMaxScaler(feature_range=(0, 1)) # Min 0, Max 1
    # Create scaler for target (Y)
    target_scaler = MinMaxScaler(feature_range=(0, 1))  # Min 0, Max 1

    # Scale feature columns
    # Learn scaling values from training data
    train_features_scaled = feature_scaler.fit_transform(
        train_data[feature_columns]
    )
    # Apply scaling to test data
    test_features_scaled = feature_scaler.transform(
        test_data[feature_columns]
    )

    # Scale target column
    train_target_scaled = target_scaler.fit_transform(
        train_data[[target_column]]
    )
    test_target_scaled = target_scaler.transform(
        test_data[[target_column]]
    )

    # Create scaled training dataframe
    scaled_train = pd.DataFrame(
        train_features_scaled,
        columns = feature_columns,
        index = train_data.index
    )

    # Add scaled target column
    scaled_train[target_column] = train_target_scaled
    # Add date column back
    scaled_train["date"] = train_data["date"]

    # Create scaled testing dataframe
    scaled_test = pd.DataFrame(
        test_features_scaled,
        columns = feature_columns,
        index = test_data.index
    )

    # Add scaled target column
    scaled_test[target_column] = test_target_scaled
    # Add date column back
    scaled_test["date"] = test_data["date"]

    # Return all outputs
    return (
        scaled_train,
        scaled_test,
        feature_scaler,
        target_scaler,
        feature_columns
    )

