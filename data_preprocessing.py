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


