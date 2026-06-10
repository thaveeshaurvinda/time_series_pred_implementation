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



